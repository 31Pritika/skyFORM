import { useCallback, useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { API, PortalContext, useRemote } from './portal/context';
import Shell from './portal/Shell';
import { Overview, Ingest, Workspace, Analytics, Exports } from './portal/Pages';
import '@fontsource/space-grotesk/400.css';
import '@fontsource/space-grotesk/500.css';
import '@fontsource/space-grotesk/600.css';
import '@fontsource/jetbrains-mono/400.css';
import './portal/portal.css';
function Application(){
 const [catalog,setCatalog]=useState({projects:[],pipeline:null}),[selected,setSelected]=useState(()=>localStorage.getItem('skyform.project')||''),[error,setError]=useState(''),[tick,setTick]=useState(0),[animate,setAnimateState]=useState(()=>localStorage.getItem('skyform.animate')!=='false');
 const refresh=useCallback(()=>setTick(n=>n+1),[]);
 useEffect(()=>{let live=true;const poll=()=>fetch(`${API}/api/projects`,{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Local engine unavailable. Start the SkyFORM backend to access projects.');return r.json();}).then(data=>{if(!live)return;setCatalog(data);setError('');setSelected(id=>id&&data.projects.some(p=>p.id===id)?id:data.pipeline?.video_id||data.projects[0]?.id||'');}).catch(e=>{if(live)setError(e.message==='Failed to fetch'?'Cannot connect to the local engine. Check that the backend is running.':e.message);});poll();const timer=setInterval(poll,4000);return()=>{live=false;clearInterval(timer);};},[tick]);
 const projectResult=useRemote(selected?`/api/projects/${selected}`:null,`${tick}-${catalog.pipeline?.status}-${catalog.pipeline?.completed_at}`);
 const select=useCallback(id=>{setSelected(id);localStorage.setItem('skyform.project',id);},[]);
 const setAnimate=v=>{setAnimateState(v);localStorage.setItem('skyform.animate',String(v));};
 return <PortalContext.Provider value={{...catalog,project:projectResult.data,projectError:projectResult.error,selected,select,refresh,error,online:!error,animate,setAnimate}}><Routes><Route element={<Shell/>}><Route index element={<Overview/>}/><Route path="ingest" element={<Ingest/>}/><Route path="workspace" element={<Workspace/>}/><Route path="analytics" element={<Analytics/>}/><Route path="export" element={<Exports/>}/><Route path="*" element={<main className="page"><div className="empty-state"><h1>Route not found</h1><p>This operation does not exist.</p><Link className="primary" to="/">Return to mission control</Link></div></main>}/></Route></Routes></PortalContext.Provider>;
}
export default function App(){return <BrowserRouter><Application/></BrowserRouter>;}
