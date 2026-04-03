import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';

/* ─── Animated Waveform ───────────────────────────────────────────────────── */

function generateWavePath(
  cycles: number,
  segmentWidth: number,
  startX: number,
  centerY: number,
  amplitude: number
): string {
  const waveLen = segmentWidth / cycles;
  const half = waveLen / 2;
  let d = `M ${startX} ${centerY}`;
  for (let i = 0; i < cycles; i++) {
    const x = startX + i * waveLen;
    const cp = waveLen * 0.3;
    d += ` C ${x + cp} ${centerY - amplitude}, ${x + half - cp} ${centerY - amplitude}, ${x + half} ${centerY}`;
    d += ` C ${x + half + cp} ${centerY + amplitude}, ${x + waveLen - cp} ${centerY + amplitude}, ${x + waveLen} ${centerY}`;
  }
  return d;
}

const WaveformHero: React.FC = () => {
  const W = 2800;
  const H = 160;
  const cy = H / 2;
  const half = W / 2;

  const makePath = (cycles: number, amplitude: number) =>
    generateWavePath(cycles, half, 0, cy, amplitude) +
    ' ' +
    generateWavePath(cycles, half, half, cy, amplitude);

  const wave1 = makePath(10, 38);
  const wave2 = makePath(10, 24);
  const wave3 = makePath(14, 14);

  return (
    <div style={{ width: '100%', overflow: 'hidden', height: H, position: 'relative', marginTop: '2.5rem' }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{ width: '200%', height: '100%', display: 'block' }}
        className="waveform-svg"
      >
        <path d={wave1} stroke="#F97316" strokeWidth="3" fill="none" opacity="0.18" />
        <path d={wave2} stroke="#F97316" strokeWidth="2.5" fill="none" opacity="0.35" />
        <path d={wave3} stroke="#F97316" strokeWidth="2" fill="none" opacity="0.6" />
      </svg>
    </div>
  );
};

/* ─── SVG Icons ───────────────────────────────────────────────────────────── */

const IconBolt = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#F97316" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);

const IconMusic = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#F97316" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 18V5l12-2v13" />
    <circle cx="6" cy="18" r="3" />
    <circle cx="18" cy="16" r="3" />
  </svg>
);

const IconGlobe = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#F97316" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="2" y1="12" x2="22" y2="12" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
  </svg>
);

const LogoWave = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#F97316" strokeWidth="2.5" strokeLinecap="round">
    <path d="M2 12 Q 5 6, 8 12 Q 11 18, 14 12 Q 17 6, 20 12 Q 22 15, 24 12" />
  </svg>
);

const StarIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="#F97316" stroke="none">
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
  </svg>
);

/* ─── Main Component ──────────────────────────────────────────────────────── */

const LandingPage: React.FC = () => {
  useEffect(() => {
    const nav = document.getElementById('landing-nav');
    const onScroll = () => {
      if (!nav) return;
      if (window.scrollY > 10) {
        nav.style.boxShadow = '0 2px 24px rgba(44,26,6,0.08)';
      } else {
        nav.style.boxShadow = 'none';
      }
    };
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }

        .landing-root {
          font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
          background: #FDF6EC;
          color: #2C1A06;
          scroll-behavior: smooth;
          overflow-x: hidden;
        }

        /* ── Navbar ── */
        .l-nav {
          position: sticky;
          top: 0;
          z-index: 100;
          height: 64px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 6vw;
          backdrop-filter: blur(18px);
          -webkit-backdrop-filter: blur(18px);
          background: rgba(253, 246, 236, 0.88);
          border-bottom: 1px solid rgba(249, 115, 22, 0.12);
          transition: box-shadow 0.3s ease;
        }
        .l-nav-logo {
          display: flex;
          align-items: center;
          gap: 8px;
          text-decoration: none;
          color: #2C1A06;
        }
        .l-nav-logo span {
          font-size: 1.15rem;
          font-weight: 800;
          letter-spacing: -0.3px;
          color: #2C1A06;
        }
        .l-beta-badge {
          font-size: 0.6rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          background: rgba(251, 191, 36, 0.18);
          color: #92400E;
          border: 1px solid rgba(180, 83, 9, 0.18);
          border-radius: 999px;
          padding: 0.18rem 0.5rem;
          align-self: center;
          margin-left: -2px;
        }
        .l-nav-right {
          display: flex;
          align-items: center;
          gap: 1.25rem;
        }
        .l-signin-link {
          font-size: 0.9rem;
          font-weight: 600;
          color: #92400E;
          text-decoration: none;
          transition: color 0.15s;
        }
        .l-signin-link:hover { color: #F97316; }

        .l-pill-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: #F97316;
          color: #fff;
          border: none;
          border-radius: 999px;
          padding: 0.5rem 1.25rem;
          font-size: 0.9rem;
          font-weight: 700;
          font-family: inherit;
          cursor: pointer;
          text-decoration: none;
          transition: background 0.18s, transform 0.18s, box-shadow 0.18s;
          box-shadow: 0 2px 12px rgba(249,115,22,0.28);
          letter-spacing: 0.01em;
        }
        .l-pill-btn:hover {
          background: #C2410C;
          transform: translateY(-1px) scale(1.03);
          box-shadow: 0 6px 20px rgba(249,115,22,0.38);
        }
        .l-pill-btn:active {
          transform: translateY(0) scale(0.99);
        }

        .l-pill-btn-lg {
          padding: 0.75rem 2rem;
          font-size: 1.05rem;
          box-shadow: 0 4px 20px rgba(249,115,22,0.35);
        }
        .l-pill-btn-lg:hover {
          transform: translateY(-2px) scale(1.04);
          box-shadow: 0 8px 28px rgba(249,115,22,0.42);
        }

        .l-pill-btn-cream {
          background: #FDF6EC;
          color: #C2410C;
          box-shadow: 0 4px 20px rgba(44,26,6,0.15);
        }
        .l-pill-btn-cream:hover {
          background: #FFF7ED;
          color: #9A3412;
          box-shadow: 0 8px 28px rgba(44,26,6,0.2);
        }

        /* ── Hero ── */
        .l-hero {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 5rem 6vw 2rem;
          background: radial-gradient(ellipse 80% 60% at 50% 40%, #FFF0D6 0%, #FDF6EC 70%);
          position: relative;
          overflow: hidden;
        }

        .l-hero-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: rgba(249,115,22,0.1);
          border: 1px solid rgba(249,115,22,0.25);
          color: #C2410C;
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          border-radius: 999px;
          padding: 0.3rem 0.9rem;
          margin-bottom: 1.75rem;
        }
        .l-hero-badge-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #F97316;
          animation: badge-pulse 2s ease-in-out infinite;
        }
        @keyframes badge-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.7); }
        }

        .l-hero-h1 {
          font-size: clamp(2.6rem, 6vw, 4.5rem);
          font-weight: 800;
          line-height: 1.08;
          letter-spacing: -0.03em;
          color: #2C1A06;
          max-width: 780px;
          margin-bottom: 1.25rem;
        }
        .l-hero-h1 em {
          font-style: normal;
          color: #F97316;
        }

        .l-hero-sub {
          font-size: clamp(1rem, 2vw, 1.2rem);
          color: #78350F;
          font-weight: 400;
          line-height: 1.65;
          max-width: 520px;
          margin-bottom: 2.5rem;
        }

        .l-hero-cta-row {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          flex-wrap: wrap;
          margin-bottom: 0.75rem;
        }

        .l-hero-note {
          font-size: 0.8rem;
          color: #B45309;
          margin-top: 0.75rem;
        }

        /* Waveform animation */
        .waveform-svg {
          animation: wave-flow 14s linear infinite;
        }
        @keyframes wave-flow {
          from { transform: translateX(0); }
          to { transform: translateX(-50%); }
        }

        /* ── Features ── */
        .l-features {
          background: #F5ECD6;
          padding: 6rem 6vw;
        }
        .l-section-label {
          text-align: center;
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: #B45309;
          margin-bottom: 0.75rem;
        }
        .l-section-title {
          text-align: center;
          font-size: clamp(1.75rem, 3.5vw, 2.6rem);
          font-weight: 800;
          letter-spacing: -0.025em;
          color: #2C1A06;
          margin-bottom: 0.75rem;
        }
        .l-section-sub {
          text-align: center;
          font-size: 1.05rem;
          color: #92400E;
          font-weight: 400;
          max-width: 500px;
          margin: 0 auto 3.5rem;
          line-height: 1.6;
        }

        .l-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
          gap: 1.5rem;
          max-width: 1060px;
          margin: 0 auto;
        }
        .l-card {
          background: #FEFAF3;
          border: 1px solid rgba(249,115,22,0.12);
          border-radius: 1.25rem;
          padding: 2rem 1.75rem;
          box-shadow: 0 2px 16px rgba(44,26,6,0.05);
          transition: transform 0.22s ease, box-shadow 0.22s ease;
        }
        .l-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 32px rgba(249,115,22,0.14);
        }
        .l-card-icon {
          width: 52px;
          height: 52px;
          background: rgba(249,115,22,0.1);
          border-radius: 0.875rem;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 1.25rem;
        }
        .l-card-title {
          font-size: 1.1rem;
          font-weight: 700;
          color: #2C1A06;
          margin-bottom: 0.5rem;
          letter-spacing: -0.01em;
        }
        .l-card-body {
          font-size: 0.95rem;
          color: #78350F;
          line-height: 1.65;
          font-weight: 400;
        }

        /* ── Social proof ── */
        .l-proof {
          background: #FDF6EC;
          padding: 4rem 6vw;
          text-align: center;
        }
        .l-stars {
          display: flex;
          justify-content: center;
          gap: 4px;
          margin-bottom: 1rem;
        }
        .l-proof-text {
          font-size: 1.05rem;
          color: #78350F;
          font-weight: 500;
        }
        .l-proof-sub {
          font-size: 0.875rem;
          color: #B45309;
          margin-top: 0.5rem;
          font-weight: 400;
        }

        /* ── Divider ── */
        .l-divider {
          height: 1px;
          background: linear-gradient(to right, transparent, rgba(249,115,22,0.2), transparent);
          margin: 0 6vw;
        }

        /* ── CTA Band ── */
        .l-cta-band {
          background: linear-gradient(135deg, #FFF0D6 0%, #FFE0B2 40%, #FFCC80 100%);
          padding: 6rem 6vw;
          text-align: center;
          position: relative;
          overflow: hidden;
        }
        .l-cta-band::before {
          content: '';
          position: absolute;
          inset: 0;
          background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23F97316' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
          pointer-events: none;
        }
        .l-cta-band-h2 {
          font-size: clamp(2rem, 4vw, 3rem);
          font-weight: 800;
          letter-spacing: -0.03em;
          color: #2C1A06;
          margin-bottom: 0.75rem;
          position: relative;
        }
        .l-cta-band-sub {
          font-size: 1.05rem;
          color: #78350F;
          font-weight: 400;
          margin-bottom: 2.5rem;
          position: relative;
        }
        .l-cta-band-btn {
          position: relative;
        }

        /* ── Footer ── */
        .l-footer {
          background: #F0E6D0;
          padding: 1.75rem 6vw;
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 0.75rem;
          border-top: 1px solid rgba(249,115,22,0.1);
        }
        .l-footer-logo {
          display: flex;
          align-items: center;
          gap: 7px;
          text-decoration: none;
          color: #2C1A06;
        }
        .l-footer-logo span {
          font-size: 0.95rem;
          font-weight: 800;
          color: #2C1A06;
        }
        .l-footer-copy {
          font-size: 0.82rem;
          color: #92400E;
          font-weight: 400;
        }
      `}</style>

      <div className="landing-root">

        {/* ── Navbar ─────────────────────────────────────────────────────── */}
        <nav className="l-nav" id="landing-nav">
          <Link to="/" className="l-nav-logo">
            <LogoWave />
            <span>Score AI</span>
            <span className="l-beta-badge">Beta</span>
          </Link>
          <div className="l-nav-right">
            <Link to="/login" className="l-signin-link">Sign In</Link>
            <Link to="/login" className="l-pill-btn">
              Sign Up &rarr;
            </Link>
          </div>
        </nav>

        {/* ── Hero ───────────────────────────────────────────────────────── */}
        <section className="l-hero">
          <div className="l-hero-badge">
            <span className="l-hero-badge-dot" />
            AI-Powered &nbsp;·&nbsp; Real Time
          </div>

          <h1 className="l-hero-h1">
            Live Music,<br /><em>Instantly</em> Readable.
          </h1>

          <p className="l-hero-sub">
            Score AI listens to any live performance and turns it into
            beautiful sheet music in seconds — no setup, no delays, no limits.
          </p>

          <div className="l-hero-cta-row">
            <Link to="/login" className="l-pill-btn l-pill-btn-lg">
              Get Started Free &rarr;
            </Link>
          </div>

          <p className="l-hero-note">No credit card required &nbsp;·&nbsp; Works in your browser</p>

          <WaveformHero />
        </section>

        {/* ── Features ───────────────────────────────────────────────────── */}
        <section className="l-features">
          <p className="l-section-label">What Score AI Does</p>
          <h2 className="l-section-title">Everything a musician needs</h2>
          <p className="l-section-sub">
            From the first note to the final measure — Score AI handles the transcription so you can focus on the music.
          </p>

          <div className="l-cards">
            <div className="l-card">
              <div className="l-card-icon"><IconBolt /></div>
              <div className="l-card-title">Real-Time Transcription</div>
              <p className="l-card-body">
                Hear a note, see a note. Score AI processes audio live with zero perceptible delay, rendering notation as you perform.
              </p>
            </div>
            <div className="l-card">
              <div className="l-card-icon"><IconMusic /></div>
              <div className="l-card-title">Any Instrument</div>
              <p className="l-card-body">
                Piano, guitar, voice, or a full ensemble — our AI model recognizes pitch across the entire audible musical range.
              </p>
            </div>
            <div className="l-card">
              <div className="l-card-icon"><IconGlobe /></div>
              <div className="l-card-title">Use It Anywhere</div>
              <p className="l-card-body">
                Browser-based and cloud-powered. Open Score AI at rehearsal, on stage, in a classroom, or from the comfort of your home.
              </p>
            </div>
          </div>
        </section>

        <div className="l-divider" />

        {/* ── Social Proof ───────────────────────────────────────────────── */}
        <section className="l-proof">
          <div className="l-stars">
            <StarIcon /><StarIcon /><StarIcon /><StarIcon /><StarIcon />
          </div>
          <p className="l-proof-text">
            Trusted by musicians, students, and music lovers worldwide
          </p>
          <p className="l-proof-sub">
            From practice rooms to concert halls — Score AI travels with you.
          </p>
        </section>

        <div className="l-divider" />

        {/* ── Closing CTA ────────────────────────────────────────────────── */}
        <section className="l-cta-band">
          <h2 className="l-cta-band-h2">Ready to read the music?</h2>
          <p className="l-cta-band-sub">
            Start transcribing in seconds. Free forever for personal use.
          </p>
          <div className="l-cta-band-btn">
            <Link to="/login" className="l-pill-btn l-pill-btn-lg l-pill-btn-cream">
              Sign Up Free &rarr;
            </Link>
          </div>
        </section>

        {/* ── Footer ─────────────────────────────────────────────────────── */}
        <footer className="l-footer">
          <Link to="/" className="l-footer-logo">
            <LogoWave />
            <span>Score AI</span>
          </Link>
          <p className="l-footer-copy">© 2026 Score AI. All rights reserved.</p>
        </footer>

      </div>
    </>
  );
};

export default LandingPage;
