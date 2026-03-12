import Link from "next/link";

export default function Home() {
  const designs = [
    { id: 1, name: "Design 1" },
    { id: 2, name: "Design 2" },
    { id: 3, name: "Design 3" },
    { id: 4, name: "Design 4" },
    { id: 5, name: "Design 5" },
    { id: 6, name: "Design 6" },
    { id: 7, name: "Design 7" },
    { id: 8, name: "Design 8" },
  ];

  return (
    <div className="min-h-screen bg-white text-gray-900 flex flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold mb-2">ExoBiome - Design Variants</h1>
      <p className="text-gray-500 mb-8">HACK-4-SAGES 2026</p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {designs.map((d) => (
          <Link
            key={d.id}
            href={`/${d.id}`}
            className="block p-8 border border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all text-center"
          >
            <span className="text-2xl font-mono">/{d.id}</span>
            <br />
            <span className="text-sm text-gray-400">{d.name}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
