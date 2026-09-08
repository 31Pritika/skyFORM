import { createContext, useContext, useEffect, useState } from 'react';
export const API = 'http://127.0.0.1:8000';
export const PortalContext = createContext(null);
export const usePortal = () => useContext(PortalContext);
export const fmt = (n, digits=0) => Number.isFinite(n) ? n.toLocaleString(undefined,{maximumFractionDigits:digits}) : '—';
export const bytes = n => n >= 1048576 ? `${(n/1048576).toFixed(1)} MB` : `${Math.ceil(n/1024)} KB`;
export function useRemote(path, refresh=0) {
  const [result,setResult] = useState({});
  useEffect(() => {
    if (!path) return;
    const controller=new AbortController();
    fetch(`${API}${path}`,{signal:controller.signal,cache:'no-store'})
      .then(async r => { const data=await r.json(); if(!r.ok) throw new Error(data.detail || 'Request failed');return data; })
      .then(data=>setResult({path,data,error:null}))
      .catch(error=>{if(error.name!=='AbortError')setResult({path,data:null,error:error.message});});
    return ()=>controller.abort();
  },[path,refresh]);
  return result.path===path ? result : {data:null,error:null};
}
