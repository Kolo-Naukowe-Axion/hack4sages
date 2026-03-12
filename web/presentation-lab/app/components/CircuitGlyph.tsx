export function CircuitGlyph() {
  return (
    <div className="figure-frame">
      <h3 className="section-title">Hybrid Circuit</h3>
      <svg viewBox="0 0 540 210" role="img" aria-label="Quantum residual diagram">
        <rect x="0" y="0" width="540" height="210" rx="22" fill="rgba(255,255,255,0.24)" />
        {Array.from({ length: 8 }, (_, index) => {
          const y = 36 + index * 18;
          return (
            <g key={index}>
              <line x1="70" x2="468" y1={y} y2={y} stroke="rgba(8,60,84,0.24)" strokeWidth="2" />
              <text x="26" y={y + 4} fontSize="11" fill="rgba(17,17,17,0.58)">
                q{index}
              </text>
            </g>
          );
        })}
        {[0, 2, 4, 6].map((index, gateIndex) => {
          const x = 130 + gateIndex * 88;
          return (
            <g key={x}>
              <rect x={x} y={27 + index * 18} width="34" height="18" rx="6" fill="#0d5f63" />
              <text x={x + 10} y={40 + index * 18} fontSize="10" fill="white">
                RY
              </text>
              <circle cx={x + 58} cy={36 + index * 18} r="7" fill="#d79a2b" />
              <line
                x1={x + 58}
                x2={x + 58}
                y1={36 + index * 18}
                y2={54 + index * 18}
                stroke="#d79a2b"
                strokeWidth="2"
              />
              <circle cx={x + 58} cy={54 + index * 18} r="9" fill="none" stroke="#d79a2b" strokeWidth="2" />
            </g>
          );
        })}
        <path
          d="M430 34 C474 34, 486 62, 486 104 C486 145, 474 173, 430 173"
          fill="none"
          stroke="rgba(125,69,65,0.75)"
          strokeWidth="3"
          strokeLinecap="round"
        />
        <text x="360" y="102" fontSize="13" fill="rgba(17,17,17,0.68)">
          gated residual correction
        </text>
      </svg>
    </div>
  );
}
