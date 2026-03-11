export function GradientMesh() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
      <div
        className="mesh-orb absolute -top-32 -left-32 w-[600px] h-[600px] rounded-full opacity-20 blur-[120px]"
        style={{ background: "radial-gradient(circle, #6366f1 0%, transparent 70%)" }}
      />
      <div
        className="mesh-orb-alt absolute top-1/3 -right-48 w-[500px] h-[500px] rounded-full opacity-15 blur-[100px]"
        style={{ background: "radial-gradient(circle, #a855f7 0%, transparent 70%)" }}
      />
      <div
        className="mesh-orb absolute bottom-0 left-1/3 w-[400px] h-[400px] rounded-full opacity-10 blur-[80px]"
        style={{ background: "radial-gradient(circle, #ec4899 0%, transparent 70%)" }}
      />
    </div>
  );
}
