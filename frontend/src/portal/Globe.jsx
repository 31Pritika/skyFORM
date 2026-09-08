import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMemo } from 'react';
import * as THREE from 'three';
function World() {
  const geometry=useMemo(()=>{
    const points=[], colors=[];
    for(let i=0;i<21000;i++){
      const y=1-2*i/20999, a=i*Math.PI*(3-Math.sqrt(5)), r=Math.sqrt(1-y*y);
      const x=Math.cos(a)*r,z=Math.sin(a)*r;
      const land=Math.sin(x*8+z*4)*Math.cos(y*6-z*3)+Math.sin(z*11+y*3)*.3;
      const radius=1.36+(land>.2?Math.max(land,0)*.027:0);
      points.push(x*radius,y*radius,z*radius);
      colors.push(...(land>.2?[.15,.85,.88]:[.08,.24,.32]));
    }
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(points,3));g.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));return g;
  },[]);
  return <group rotation={[.1,0,.22]}><mesh><sphereGeometry args={[1.33,48,32]}/><meshBasicMaterial color="#080d14"/></mesh><points geometry={geometry}><pointsMaterial size={.009} vertexColors transparent opacity={.88}/></points><mesh><sphereGeometry args={[1.365,24,16]}/><meshBasicMaterial color="#193140" wireframe transparent opacity={.27}/></mesh><mesh rotation={[Math.PI/2,0,0]}><torusGeometry args={[1.65,.002,4,160]}/><meshBasicMaterial color="#246573"/></mesh></group>;
}
export default function Globe({rotate=true}) { return <Canvas camera={{position:[0,.3,3.8],fov:45}} dpr={[1,1.5]}><color attach="background" args={['#080b11']}/><World/><OrbitControls enablePan={false} enableZoom autoRotate={rotate} autoRotateSpeed={.35} minDistance={2.5} maxDistance={6}/></Canvas>; }
