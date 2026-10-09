const fs = require('fs');
const path = require('path');

// Locate resvg from ecosystem node_modules
let Resvg;
const possiblePaths = [
  'B:/workgit/mtlg-site/node_modules/@resvg/resvg-js',
  'B:/workgit/dnswatch/ui/node_modules/@resvg/resvg-js',
  path.join(__dirname, '../node_modules/@resvg/resvg-js'),
];

for (const p of possiblePaths) {
  try {
    Resvg = require(p).Resvg;
    break;
  } catch (e) {}
}

if (!Resvg) {
  console.error('Could not locate @resvg/resvg-js in ecosystem.');
  process.exit(1);
}

function buildLogoSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
  <defs>
    <clipPath id="squircle-clip">
      <rect x="24" y="24" width="976" height="976" rx="220" />
    </clipPath>

    <linearGradient id="cyan-glow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00f5ff"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>

    <linearGradient id="amber-glow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#fbbf24"/>
      <stop offset="100%" stop-color="#ea580c"/>
    </linearGradient>

    <linearGradient id="horn-grad-left" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="60%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>

    <linearGradient id="horn-grad-right" x1="100%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00f5ff"/>
      <stop offset="60%" stop-color="#0369a1"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>

    <filter id="subtle-shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="16" stdDeviation="20" flood-color="#0f172a" flood-opacity="0.15" />
    </filter>
  </defs>

  <!-- Luxury White Squircle Container -->
  <rect x="24" y="24" width="976" height="976" rx="220" fill="#ffffff" stroke="#e2e8f0" stroke-width="6" />

  <g clip-path="url(#squircle-clip)">
    <g transform="translate(512, 512)" filter="url(#subtle-shadow)">

      <!-- Hexagonal Architectural Gateway Frame -->
      <polygon points="
        0,-390
        338,-195
        338,195
        0,390
        -338,195
        -338,-195
      " fill="none" stroke="#0f172a" stroke-width="36" stroke-linejoin="round" />

      <polygon points="
        0,-355
        307,-177
        307,177
        0,355
        -307,177
        -307,-177
      " fill="none" stroke="#00f5ff" stroke-width="4" opacity="0.45" stroke-dasharray="16, 12" />

      <!-- Corner Telemetry Nodes -->
      <circle cx="0" cy="-390" r="12" fill="#00f5ff" />
      <circle cx="338" cy="-195" r="10" fill="#0f172a" stroke="#00f5ff" stroke-width="3" />
      <circle cx="338" cy="195" r="10" fill="#0f172a" stroke="#00f5ff" stroke-width="3" />
      <circle cx="0" cy="390" r="12" fill="#00f5ff" />
      <circle cx="-338" cy="195" r="10" fill="#0f172a" stroke="#00f5ff" stroke-width="3" />
      <circle cx="-338" cy="-195" r="10" fill="#0f172a" stroke="#00f5ff" stroke-width="3" />

      <!-- ========================================================= -->
      <!-- THE VULCAN FORGE RAM (COMPRESSIVE LOW-RANK MATRIX MASCOT) -->
      <!-- ========================================================= -->

      <!-- 1. Watertight Solid Base Silhouette (Zero Background Leaks) -->
      <path d="
        M 0 -290
        L 90 -280 L 190 -250 L 290 -160 L 320 -50 L 290 60 L 220 120 L 170 80
        L 210 20 L 230 -60 L 180 -130 L 110 -150 L 140 -50 L 110 90 L 60 210
        L 45 290 L 0 320 L -45 290 L -60 210 L -110 90 L -140 -50 L -110 -150
        L -180 -130 L -230 -60 L -210 20 L -170 80 L -220 120 L -290 60
        L -320 -50 L -290 -160 L -190 -250 L -90 -280 Z
      " fill="#090d16" />

      <!-- 2. Twin Sweeping Spiral Matrix Horns (Adapter Decomposition A x B) -->
      <!-- RIGHT HORN (Viewer Right) -->
      <!-- Outer Curve Segments -->
      <polygon points="90,-280 190,-250 140,-200 65,-220" fill="#1e293b" />
      <polygon points="190,-250 290,-160 220,-130 140,-200" fill="#334155" />
      <polygon points="290,-160 320,-50 250,-40 220,-130" fill="#1e293b" />
      <polygon points="320,-50 290,60 230,20 250,-40" fill="#0f172a" />
      <polygon points="290,60 220,120 180,50 230,20" fill="#334155" />
      <polygon points="220,120 170,80 180,50" fill="#1e293b" />
      
      <!-- Inner Horn Curl Facets (Looping back) -->
      <polygon points="230,20 180,50 210,20" fill="url(#horn-grad-right)" opacity="0.9" />
      <polygon points="250,-40 230,20 210,20 230,-60" fill="#0284c7" />
      <polygon points="220,-130 250,-40 230,-60 180,-130" fill="#0369a1" />
      <polygon points="140,-200 220,-130 180,-130 110,-150" fill="#00f5ff" opacity="0.3" />
      <polygon points="65,-220 140,-200 110,-150 50,-170" fill="#1e293b" />

      <!-- LEFT HORN (Viewer Left - Symmetrical Mirror) -->
      <!-- Outer Curve Segments -->
      <polygon points="-90,-280 -190,-250 -140,-200 -65,-220" fill="#0f172a" />
      <polygon points="-190,-250 -290,-160 -220,-130 -140,-200" fill="#1e293b" />
      <polygon points="-290,-160 -320,-50 -250,-40 -220,-130" fill="#0b0f19" />
      <polygon points="-320,-50 -290,60 -230,20 -250,-40" fill="#1e293b" />
      <polygon points="-290,60 -220,120 -180,50 -230,20" fill="#0f172a" />
      <polygon points="-220,120 -170,80 -180,50" fill="#090d16" />

      <!-- Inner Horn Curl Facets -->
      <polygon points="-230,20 -180,50 -210,20" fill="url(#horn-grad-left)" opacity="0.85" />
      <polygon points="-250,-40 -230,20 -210,20 -230,-60" fill="#0369a1" />
      <polygon points="-220,-130 -250,-40 -230,-60 -180,-130" fill="#0284c7" />
      <polygon points="-140,-200 -220,-130 -180,-130 -110,-150" fill="#00f5ff" opacity="0.25" />
      <polygon points="-65,-220 -140,-200 -110,-150 -50,-170" fill="#0f172a" />

      <!-- 3. Cranial Crown & Anvil Forehead (Rigid Low-Poly Plate) -->
      <polygon points="0,-290 90,-280 65,-220 0,-230" fill="#334155" />
      <polygon points="0,-290 -90,-280 -65,-220 0,-230" fill="#1e293b" />

      <polygon points="0,-230 65,-220 50,-170 0,-160" fill="#475569" />
      <polygon points="0,-230 -65,-220 -50,-170 0,-160" fill="#334155" />

      <!-- Central Matrix Nexus Diamond (Forehead Armor Plate) -->
      <polygon points="0,-160 50,-170 70,-90 0,-70" fill="#1e293b" />
      <polygon points="0,-160 -50,-170 -70,-90 0,-70" fill="#0f172a" />

      <!-- Forehead Chevron Inset (Forge Core Symbolism) -->
      <polygon points="0,-130 35,-140 0,-85" fill="#00f5ff" opacity="0.9" />
      <polygon points="0,-130 -35,-140 0,-85" fill="#0284c7" opacity="0.85" />

      <!-- 4. Mid Face & Brow Ridge -->
      <polygon points="0,-70 70,-90 100,-40 0,-30" fill="#334155" />
      <polygon points="0,-70 -70,-90 -100,-40 0,-30" fill="#1e293b" />

      <polygon points="70,-90 110,-150 140,-50 100,-40" fill="#1e293b" />
      <polygon points="-70,-90 -110,-150 -140,-50 -100,-40" fill="#0f172a" />

      <!-- 5. Electric Cyan Predator Optics (Alert Gaze Principle) -->
      <!-- Right Eye (Viewer Right) -->
      <polygon points="40,-35 110,-45 85,-15 40,-22" fill="#090d16" stroke="#0f172a" stroke-width="2" />
      <polygon points="48,-32 100,-40 80,-18 50,-24" fill="url(#cyan-glow)" />
      <!-- Slit Pupil -->
      <polygon points="70,-36 78,-38 74,-20 68,-21" fill="#090d16" />

      <!-- Left Eye (Viewer Left) -->
      <polygon points="-40,-35 -110,-45 -85,-15 -40,-22" fill="#090d16" stroke="#0f172a" stroke-width="2" />
      <polygon points="-48,-32 -100,-40 -80,-18 -50,-24" fill="url(#cyan-glow)" />
      <!-- Slit Pupil -->
      <polygon points="-70,-36 -78,-38 -74,-20 -68,-21" fill="#090d16" />

      <!-- Under-Eye Cheekbone Facets -->
      <polygon points="0,-30 40,-22 85,-15 60,50 0,40" fill="#1e293b" />
      <polygon points="0,-30 -40,-22 -85,-15 -60,50 0,40" fill="#0f172a" />

      <polygon points="85,-15 100,-40 140,-50 110,90 60,50" fill="#334155" />
      <polygon points="-85,-15 -100,-40 -140,-50 -110,90 -60,50" fill="#1e293b" />

      <!-- 6. Bridge of Muzzle & Nasal Ridge -->
      <polygon points="0,40 60,50 45,150 0,160" fill="#334155" />
      <polygon points="0,40 -60,50 -45,150 0,160" fill="#1e293b" />

      <polygon points="60,50 110,90 60,210 45,150" fill="#1e293b" />
      <polygon points="-60,50 -110,90 -60,210 -45,150" fill="#0f172a" />

      <!-- 7. Muzzle & Chiseled Jaw (Zero Buck Teeth - Natural Obsidian Shading) -->
      <polygon points="0,160 45,150 35,230 0,240" fill="#0f172a" />
      <polygon points="0,160 -45,150 -35,230 0,240" fill="#090d16" />

      <!-- Nose Tip & Nostrils -->
      <polygon points="0,210 25,200 20,240 0,245" fill="#090d16" />
      <polygon points="0,210 -25,200 -20,240 0,245" fill="#090d16" />
      <!-- Subtle Cyan Nostril Telemetry Slits -->
      <line x1="6" y1="228" x2="16" y2="225" stroke="#00f5ff" stroke-width="2.5" stroke-linecap="round" />
      <line x1="-6" y1="228" x2="-16" y2="225" stroke="#00f5ff" stroke-width="2.5" stroke-linecap="round" />

      <!-- Lower Jaw / Beard Keel Plate -->
      <polygon points="0,245 20,240 35,230 45,290 0,320" fill="#1e293b" />
      <polygon points="0,245 -20,240 -35,230 -45,290 0,320" fill="#0f172a" />

      <!-- Lateral Jaw Wings -->
      <polygon points="35,230 60,210 45,290" fill="#334155" />
      <polygon points="-35,230 -60,210 -45,290" fill="#1e293b" />

      <!-- Matrix Vector Inset on Chin (Delta W / Rank Symbol) -->
      <polygon points="0,265 15,255 0,305" fill="#00f5ff" opacity="0.8" />
      <polygon points="0,265 -15,255 0,305" fill="#0284c7" opacity="0.75" />

    </g>
  </g>
</svg>`;
}

async function renderLogo() {
  const outputDir = path.join(__dirname, '../docs/images');
  fs.mkdirSync(outputDir, { recursive: true });

  const svg = buildLogoSvg();
  const svgPath = path.join(outputDir, 'logo.svg');
  const pngPath = path.join(outputDir, 'logo.png');

  fs.writeFileSync(svgPath, svg, 'utf-8');
  console.log('✓ Wrote ' + svgPath);

  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: 1024 },
    font: { loadSystemFonts: false }
  });
  const pngData = resvg.render().asPng();
  fs.writeFileSync(pngPath, pngData);
  console.log('✓ Rendered ' + pngPath + ' (1024x1024, ' + pngData.length + ' bytes)');
}

renderLogo().catch(err => {
  console.error(err);
  process.exit(1);
});
