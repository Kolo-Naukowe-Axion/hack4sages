"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Save, Plus, Scissors, Copy, ClipboardPaste, ArrowUp, ArrowDown, Play, Square, RotateCcw, ChevronDown } from "lucide-react";

const menuItems = ["File", "Edit", "View", "Cell", "Kernel", "Help"];

export default function NotebookToolbar() {
  const pathname = usePathname();

  const navTabs = [
    { label: "ExoBiome_Analysis.ipynb", href: "/" },
    { label: "Planet_Explorer.ipynb", href: "/explorer" },
    { label: "Model_Comparison.ipynb", href: "/models" },
  ];

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-nb-toolbar border-b border-nb-toolbar-border">
      {/* Menu bar */}
      <div className="flex items-center gap-0 px-2 h-[30px] border-b border-nb-toolbar-border text-[13px]">
        <div className="flex items-center gap-0">
          {menuItems.map((item) => (
            <button
              key={item}
              className="px-2.5 py-0.5 text-nb-text hover:bg-nb-hover rounded-sm cursor-default"
              tabIndex={-1}
            >
              {item}
            </button>
          ))}
        </div>
      </div>

      {/* Icon toolbar */}
      <div className="flex items-center gap-0.5 px-2 h-[38px] border-b border-nb-toolbar-border">
        <ToolbarButton icon={<Save size={15} />} />
        <ToolbarButton icon={<Plus size={15} />} />
        <ToolbarButton icon={<Scissors size={15} />} />
        <ToolbarButton icon={<Copy size={15} />} />
        <ToolbarButton icon={<ClipboardPaste size={15} />} />
        <ToolbarDivider />
        <ToolbarButton icon={<ArrowUp size={15} />} />
        <ToolbarButton icon={<ArrowDown size={15} />} />
        <ToolbarDivider />
        <ToolbarButton icon={<Play size={15} />} highlight />
        <ToolbarButton icon={<Square size={15} />} />
        <ToolbarButton icon={<RotateCcw size={15} />} />
        <ToolbarDivider />
        <div className="flex items-center gap-1 px-2 py-1 border border-nb-border rounded text-[12px] font-mono bg-white min-w-[100px]">
          <span>Code</span>
          <ChevronDown size={12} className="ml-auto text-nb-muted" />
        </div>
      </div>

      {/* Notebook tabs */}
      <div className="flex items-center px-2 h-[30px] bg-nb-bg text-[12px] gap-0">
        {navTabs.map((tab) => {
          const active = pathname === tab.href;
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={`px-3 py-1 border-t-2 ${
                active
                  ? "bg-nb-cell border-t-nb-blue text-nb-text font-medium"
                  : "bg-nb-bg border-t-transparent text-nb-muted hover:bg-white hover:text-nb-text"
              } border-x border-nb-border transition-colors`}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>
    </div>
  );
}

function ToolbarButton({ icon, highlight }: { icon: React.ReactNode; highlight?: boolean }) {
  return (
    <button
      className={`p-1.5 rounded hover:bg-nb-hover ${highlight ? "text-nb-green" : "text-nb-muted"}`}
      tabIndex={-1}
    >
      {icon}
    </button>
  );
}

function ToolbarDivider() {
  return <div className="w-px h-5 bg-nb-border mx-1" />;
}
