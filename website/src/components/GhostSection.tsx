interface GhostSectionProps {
  id: string;
  index: string;
  eyebrow: string;
  title: string;
  summary: string;
  children: React.ReactNode;
}

export function GhostSection({
  id,
  index,
  eyebrow,
  title,
  summary,
  children,
}: GhostSectionProps) {
  return (
    <section id={id} className="paper-section section-anchor">
      <div className="section-head">
        <div className="section-kicker">
          <span>{index}</span>
          <span>{eyebrow}</span>
        </div>
        <div className="section-heading-block">
          <h2 className="section-title">{title}</h2>
          <p className="section-summary">{summary}</p>
        </div>
      </div>

      <div className="section-content">{children}</div>
    </section>
  );
}
