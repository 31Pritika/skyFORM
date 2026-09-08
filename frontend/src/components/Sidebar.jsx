import {
  Box,
  FolderKanban,
  ScanLine,
  History,
  Download,
  Settings,
  HelpCircle
} from "lucide-react";

const navigation = [
  { name: "Overview", icon: Box, active: true },
  { name: "Projects", icon: FolderKanban },
  { name: "New Reconstruction", icon: ScanLine },
  { name: "History", icon: History },
  { name: "Exports", icon: Download }
];

export default function Sidebar({ exportUrl }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">S</div>

        <div>
          <div className="brand-name">SkyFORM</div>
          <div className="brand-subtitle">RECONSTRUCTION SYSTEM</div>
        </div>
      </div>

      <nav className="navigation">
        <div className="nav-label">WORKSPACE</div>

        {navigation.map((item) => {
          const Icon = item.icon;

          if (item.name === "Exports" && exportUrl) {
            return <a key={item.name} className="nav-item" href={exportUrl} download>
              <Icon size={17} strokeWidth={1.7} /><span>Exports</span>
            </a>;
          }
          return (
            <button
              key={item.name}
              disabled={item.name === "Exports" && !exportUrl}
              className={`nav-item ${item.active ? "active" : ""}`}
            >
              <Icon size={17} strokeWidth={1.7} />
              <span>{item.name}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-bottom">
        <button className="nav-item">
          <HelpCircle size={17} />
          <span>Documentation</span>
        </button>

        <button className="nav-item">
          <Settings size={17} />
          <span>Settings</span>
        </button>

        <div className="system-state">
          <span className="status-dot"></span>

          <div>
            <strong>System operational</strong>
            <span>Local processing</span>
          </div>
        </div>
      </div>
    </aside>
  );
}