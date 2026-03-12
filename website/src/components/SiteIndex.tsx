"use client";

import { useEffect, useMemo, useState } from "react";
import type { SectionLink } from "@/lib/types";

interface SiteIndexProps {
  sections: SectionLink[];
}

export function SiteIndex({ sections }: SiteIndexProps) {
  const [active, setActive] = useState(sections[0]?.id ?? "");
  const ids = useMemo(() => sections.map((section) => section.id), [sections]);

  useEffect(() => {
    const observers = ids.map((id) => {
      const node = document.getElementById(id);
      if (!node) {
        return null;
      }

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              setActive(id);
            }
          });
        },
        { rootMargin: "-30% 0px -55% 0px", threshold: 0.1 },
      );

      observer.observe(node);
      return observer;
    });

    return () => {
      observers.forEach((observer) => observer?.disconnect());
    };
  }, [ids]);

  return (
    <nav className="site-index" aria-label="Page sections">
      {sections.map((section) => (
        <button
          key={section.id}
          type="button"
          className={`site-index-link ${active === section.id ? "site-index-link-active" : ""}`}
          onClick={() =>
            document.getElementById(section.id)?.scrollIntoView({
              behavior: "smooth",
              block: "start",
            })
          }
        >
          <span>{section.index}</span>
          <span>{section.label}</span>
        </button>
      ))}
    </nav>
  );
}
