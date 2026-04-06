import jsPDF from 'jspdf';

/**
 * Exports the current VexFlow sheet music to a PDF by:
 * 1. Grabbing the already-rendered SVG from the DOM
 * 2. Drawing it onto a hi-DPI canvas
 * 3. Slicing the canvas across A4 pages
 * 4. Saving with jsPDF — no backend or Lilypond required
 */
export const exportToPDF = async (title = 'sheet-music'): Promise<void> => {
  const rendererDiv = document.querySelector('[data-sheet-svg="true"]') as HTMLDivElement | null;
  const svg = rendererDiv?.querySelector('svg');

  if (!svg) {
    alert('Nothing to export — record some notes first.');
    return;
  }

  // ── 1. Determine SVG natural dimensions ─────────────────────────────────
  const svgW = parseFloat(svg.getAttribute('width')  || String(svg.viewBox.baseVal.width)  || '800');
  const svgH = parseFloat(svg.getAttribute('height') || String(svg.viewBox.baseVal.height) || '600');

  if (!svgW || !svgH) {
    alert('Could not read sheet music dimensions.');
    return;
  }

  // ── 2. Serialize SVG with explicit white background ──────────────────────
  const clone = svg.cloneNode(true) as SVGElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');

  // Prepend a white rect so the PDF background is never transparent
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  bg.setAttribute('width',  String(svgW));
  bg.setAttribute('height', String(svgH));
  bg.setAttribute('fill', '#ffffff');
  clone.insertBefore(bg, clone.firstChild);

  const svgString = new XMLSerializer().serializeToString(clone);
  const svgBlob   = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  const svgUrl    = URL.createObjectURL(svgBlob);

  // ── 3. Render SVG onto a hi-DPI canvas ───────────────────────────────────
  await new Promise<void>((resolve, reject) => {
    const img = new Image();

    img.onload = () => {
      const DPR = 2; // 2× for crisp output
      const canvas = document.createElement('canvas');
      canvas.width  = svgW * DPR;
      canvas.height = svgH * DPR;

      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.scale(DPR, DPR);
      ctx.drawImage(img, 0, 0, svgW, svgH);
      URL.revokeObjectURL(svgUrl);

      // ── 4. Build the PDF, slicing across A4 pages ─────────────────────
      //   A4: 210 × 297 mm, margin 12 mm each side
      const PAGE_W_MM  = 210;
      const PAGE_H_MM  = 297;
      const MARGIN_MM  = 12;
      const CONTENT_W_MM = PAGE_W_MM - MARGIN_MM * 2;   // 186 mm

      // px → mm: 1 px = 0.264583 mm at 96 dpi
      const PX_TO_MM = 0.264583;

      // Scale so sheet width fills the content area
      const scaleMM   = CONTENT_W_MM / (svgW * PX_TO_MM);
      const contentH_MM = svgH * PX_TO_MM * scaleMM;

      const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

      const pageContentH_MM = PAGE_H_MM - MARGIN_MM * 2;

      if (contentH_MM <= pageContentH_MM) {
        // ── Single page ──────────────────────────────────────────────────
        pdf.addImage(
          canvas.toDataURL('image/png'), 'PNG',
          MARGIN_MM, MARGIN_MM,
          CONTENT_W_MM, contentH_MM
        );
      } else {
        // ── Multi-page: slice canvas vertically per page ─────────────────
        // How many source pixels fit in one page height?
        const pxPerPage = (pageContentH_MM / scaleMM / PX_TO_MM);

        let yPx    = 0;
        let pageIdx = 0;

        while (yPx < svgH) {
          if (pageIdx > 0) pdf.addPage();

          const sliceH_px = Math.min(pxPerPage, svgH - yPx);

          const slice = document.createElement('canvas');
          slice.width  = svgW * DPR;
          slice.height = Math.ceil(sliceH_px * DPR);

          const sCtx = slice.getContext('2d')!;
          sCtx.fillStyle = '#ffffff';
          sCtx.fillRect(0, 0, slice.width, slice.height);
          sCtx.drawImage(
            canvas,
            0, Math.round(yPx * DPR),   // source top-left
            canvas.width, Math.ceil(sliceH_px * DPR),
            0, 0,
            slice.width, slice.height
          );

          const sliceH_MM = sliceH_px * PX_TO_MM * scaleMM;
          pdf.addImage(
            slice.toDataURL('image/png'), 'PNG',
            MARGIN_MM, MARGIN_MM,
            CONTENT_W_MM, sliceH_MM
          );

          yPx += pxPerPage;
          pageIdx++;
        }
      }

      // Add title as PDF metadata
      pdf.setProperties({ title });
      pdf.save(`${title}.pdf`);
      resolve();
    };

    img.onerror = () => {
      URL.revokeObjectURL(svgUrl);
      reject(new Error('Failed to load SVG for export.'));
    };

    img.src = svgUrl;
  });
};
