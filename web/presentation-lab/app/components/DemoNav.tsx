import Link from "next/link";
import { routeVariants } from "@/app/lib/project-data";

export function DemoNav() {
  return (
    <nav className="topnav" aria-label="Design variants">
      <Link href="/">Index</Link>
      {routeVariants.map((route) => (
        <Link key={route.slug} href={`/${route.slug}`}>
          {route.slug}
        </Link>
      ))}
    </nav>
  );
}
