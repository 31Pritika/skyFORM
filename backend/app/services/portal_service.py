"""Read-only project catalog, analytic calculations and deliverable downloads."""
import json
import math
from pathlib import Path
from urllib.parse import quote
from functools import lru_cache
import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.services.camera_service import normalize_pose_data
from app.services.quality_service import load_sparse_confidence
from app.services.video_service import analyze_video


def read_json(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


def job_path(base, job_id):
    if not job_id or Path(job_id).name != job_id or '..' in job_id:
        raise HTTPException(400, 'Invalid project identifier')
    path=base/'outputs/jobs'/job_id
    if not path.exists() and not (base/'outputs/videos'/f'{job_id}.json').exists():
        raise HTTPException(404, 'Project not found')
    return path


def detail(base, job_id):
    job=job_path(base,job_id); report=read_json(job/'final/pipeline_report.json')
    info=read_json(job/'final/reconstruction_info.json')
    state=read_json(base/'outputs/pipeline_state.json')
    metadata=read_json(base/'outputs/videos'/f'{job_id}.json')
    videos=sorted((base/'uploads').glob(f'{job_id}.*'))
    if not videos: videos=sorted((job/'input').glob('*.mp4'))
    video=videos[0] if videos else None
    if video and not metadata:
        try: metadata=analyze_video(str(video))
        except Exception: metadata={}
    assets=[]
    for path in sorted((job/'final').glob('*')):
        if path.is_file() and path.suffix in {'.ply','.obj','.las','.tif','.fbx','.glb','.gltf','.zip','.json','.bin','.pdf'}:
            assets.append({'name':path.name,'bytes':path.stat().st_size,'extension':path.suffix[1:],
                           'url':f'/api/projects/{quote(job_id)}/assets/{quote(path.name)}'})
    poses=normalize_pose_data(read_json(job/'colmap_poses/colmap_poses.json'))
    manifest=read_json(job/'frames/frames.json')
    times={p['image']:p['timestamp'] for p in manifest.get('frames',[])}
    for p in poses:
        p['timestamp']=times.get(p['image'])
        p['image_url']=f'/outputs/jobs/{quote(job_id)}/frames/{quote(p["image"])}'
    ready=(job/'final/skyform_pointcloud.ply').exists()
    running=state.get('video_id')==job_id and state.get('status')=='running'
    return {'id':job_id,'name':metadata.get('original_filename') or f'Reconstruction {job_id[:8]}',
            'status':'running' if running else 'ready' if ready else 'failed' if state.get('video_id')==job_id and state.get('status')=='failed' else 'uploaded',
            'video':metadata,'video_url':f'/api/projects/{job_id}/video' if video else None,
            'report':report,'info':info,'assets':assets,'cameras':poses,
            'coordinate_frame':'metric' if (job/'final/georeferencing.json').exists() else 'relative',
            'alignment':read_json(job/'final/georeferencing.json'),
            'updated_at':(job.stat().st_mtime if job.exists() else 0)}


@lru_cache(maxsize=8)
def raster_grid(path, modified):
    with rasterio.open(path) as ds:
        data=ds.read(1,masked=True)
        preview=ds.read(1,out_shape=(96,96),masked=True)
        values=data.compressed().astype(float)
        if not len(values): raise ValueError('Surface raster has no observed cells')
        area=abs(ds.transform.a*ds.transform.e)
        return {'grid':data, 'values':values, 'area':area,
                'preview':[None if masked else float(value) for value,masked in zip(preview.data.ravel(),np.ma.getmaskarray(preview).ravel())],
                'crs':ds.crs, 'transform':ds.transform, 'shape':data.shape}


def surface(job):
    paths=[job/'final/skyform_surface_georef.tif',job/'final/skyform_surface_local.tif']
    path=next((p for p in paths if p.exists()),None)
    if path is None: raise HTTPException(409,'A surface raster is required. Run reconstruction and export first.')
    return raster_grid(str(path),path.stat().st_mtime), 'm' if path.name.endswith('_georef.tif') else 'u'


def render_pdf(base, job_id, path):
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    fonts=base/'assets/fonts'
    for name,file in [('Space','SpaceGrotesk.ttf'),('Mono','JetBrainsMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    data=detail(base,job_id); report=data['report']; sfm=report.get('structure_from_motion',{}); dense=report.get('dense_reconstruction',{})
    c=canvas.Canvas(str(path),pagesize=(595,842)); c.setTitle('SkyFORM | Reconstruction summary')
    c.setFillColor(HexColor('#08090C')); c.rect(0,0,595,842,fill=1,stroke=0)
    c.setFillColor(HexColor('#00E5FF')); c.setFont('Mono',10); c.drawString(42,793,'SKYFORM / INTELLIGENCE HUB')
    c.setFillColor(HexColor('#F0F3F8')); c.setFont('Space',26); c.drawString(42,743,'Reconstruction summary')
    c.setFont('Mono',9); c.setFillColor(HexColor('#9BA7BA')); c.drawString(42,719,f'PROJECT {job_id}')
    rows=[('Coordinate frame', 'Metric aligned' if data['coordinate_frame']=='metric' else 'Relative / unscaled'),
          ('Registered cameras',str(sfm.get('registered_frames','Unavailable'))),
          ('Dense points',f'{dense.get("final_point_count",0):,}'),
          ('Processing time',str(report.get('runtime',{}).get('formatted','Unavailable'))),
          ('Surface status','Experimental; completeness not established'),
          ('Spatial accuracy','Independent validation required'),
          ('Telemetry', 'Available' if report.get('georeferencing',{}).get('telemetry_available') else 'Not supplied')]
    y=657
    for label,value in rows:
        c.setStrokeColor(HexColor('#1E2230'));c.line(42,y-17,553,y-17)
        c.setFillColor(HexColor('#9BA7BA'));c.setFont('Space',11);c.drawString(42,y,label)
        c.setFillColor(HexColor('#F0F3F8'));c.setFont('Mono',8);c.drawString(228,y,value);y-=43
    c.setFillColor(HexColor('#00E5FF')); c.setFont('Mono',10); c.drawString(42,310,'DELIVERABLES')
    y=286
    for name in [a['name'] for a in data['assets'] if a['extension'] in {'obj','ply','las','tif','glb','gltf','fbx'}][:10]:
        c.setFillColor(HexColor('#CBD4E1'));c.setFont('Mono',8);c.drawString(42,y,name);y-=17
    c.setFillColor(HexColor('#9BA7BA'));c.setFont('Space',9)
    for y,line in [(91,'Local raster values are not geographic elevations without alignment.'),(76,'Coverage, survey accuracy and long-video runtime require independent evidence.')]: c.drawString(42,y,line)
    c.setFont('Mono',8);c.drawString(42,38,'LOCAL PROCESSING / SIH 2026 / PS 158');c.drawRightString(553,38,'01 / 01')
    c.save()


def create_portal_router(base):
    router=APIRouter()
    @router.get('/api/projects')
    def catalog():
        state=read_json(base/'outputs/pipeline_state.json')
        ids={p.name for p in (base/'outputs/jobs').glob('*') if (p/'final/pipeline_report.json').exists()}
        ids|={p.stem for p in (base/'outputs/videos').glob('*.json')}
        if state.get('video_id'): ids.add(state['video_id'])
        items=[]
        for id in ids:
            job=base/'outputs/jobs'/id; r=read_json(job/'final/pipeline_report.json'); m=read_json(base/'outputs/videos'/f'{id}.json')
            items.append({'id':id,'name':m.get('original_filename') or f'Reconstruction {id[:8]}','ready':(job/'final/skyform_pointcloud.ply').exists(),
                          'points':r.get('dense_reconstruction',{}).get('final_point_count',0),'updated_at':job.stat().st_mtime if job.exists() else (base/'outputs/videos'/f'{id}.json').stat().st_mtime})
        return {'projects':sorted(items,key=lambda p:p['updated_at'],reverse=True),'pipeline':state}
    @router.get('/api/projects/{job_id}')
    def project(job_id:str): return detail(base,job_id)
    @router.get('/api/projects/{job_id}/quality')
    def quality(job_id:str):
        points=load_sparse_confidence(job_path(base,job_id))
        return {'points':points,'summary':{'point_count':len(points),'average_reprojection_error':sum(p['reprojection_error'] for p in points)/len(points) if points else None,
                                         'confidence_note':'Sparse COLMAP support, not dense-surface accuracy.'}}
    @router.get('/api/projects/{job_id}/cameras')
    def cameras(job_id:str): return {'cameras':detail(base,job_id)['cameras']}
    @router.get('/api/projects/{job_id}/video')
    def video(job_id:str):
        job=job_path(base,job_id); paths=list((base/'uploads').glob(f'{job_id}.*')) or list((job/'input').glob('*.mp4'))
        if not paths: raise HTTPException(404,'Source video unavailable')
        return FileResponse(paths[0],media_type='video/mp4')
    @router.get('/api/projects/{job_id}/assets/{name}')
    def asset(job_id:str,name:str):
        job=job_path(base,job_id)
        if Path(name).name!=name: raise HTTPException(400,'Invalid asset name')
        path=job/'final'/name
        if not path.is_file(): raise HTTPException(404,'Asset unavailable')
        return FileResponse(path,filename=name)
    @router.get('/api/projects/{job_id}/report.pdf')
    def pdf(job_id:str):
        job=job_path(base,job_id)
        if not (job/'final/pipeline_report.json').exists(): raise HTTPException(409,'Complete reconstruction before generating a report')
        path=job/'final/skyform_summary.pdf';render_pdf(base,job_id,path)
        return FileResponse(path,media_type='application/pdf',filename='SkyFORM_Summary.pdf')
    @router.get('/api/projects/{job_id}/surface')
    def raster(job_id:str):
        g,unit=surface(job_path(base,job_id));v=g['values']; hist,edges=np.histogram(v,bins=24)
        return {'width':96,'height':96,'values':g['preview'],'min':float(v.min()),'max':float(v.max()),'observed_area':len(v)*g['area'],'unit':unit,
                'histogram':hist.tolist(),'edges':edges.tolist(),'observed_cells':len(v)}
    @router.get('/api/projects/{job_id}/analytics')
    def analytics(job_id:str,level:float=0):
        if not math.isfinite(level): raise HTTPException(400,'Level must be finite')
        g,unit=surface(job_path(base,job_id));v=g['values'];a=g['area']
        return {'unit':unit,'level':level,'flooded_area':float((v<=level).sum()*a),'observed_area':float(len(v)*a),
                'cut_volume':float(np.maximum(v-level,0).sum()*a),'fill_volume':float(np.maximum(level-v,0).sum()*a),
                'note':'Vertical comparison to a constant plane over observed cells only; not a hydraulic simulation.'}
    @router.get('/api/projects/{job_id}/compare/{other_id}')
    def compare(job_id:str,other_id:str):
        a,ua=surface(job_path(base,job_id));b,ub=surface(job_path(base,other_id))
        if job_id==other_id: raise HTTPException(400,'Select two different surveys')
        if ua!='m' or ub!='m' or a['crs']!=b['crs']: raise HTTPException(409,'Both surveys need a shared metric coordinate system before spatial change detection.')
        from rasterio.warp import reproject,Resampling
        target=np.full(a['shape'],np.nan,dtype=np.float32)
        reproject(b['grid'].filled(np.nan),target,src_transform=b['transform'],src_crs=b['crs'],dst_transform=a['transform'],dst_crs=a['crs'],src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear)
        valid=~np.ma.getmaskarray(a['grid']) & np.isfinite(target)
        delta=target[valid]-a['grid'].data[valid]
        if not len(delta): raise HTTPException(409,'The surveys have no observed spatial overlap.')
        return {'overlap_area':len(delta)*a['area'],'mean_change':float(delta.mean()),'rmse':float(np.sqrt(np.mean(delta**2))),'min':float(delta.min()),'max':float(delta.max())}
    return router
