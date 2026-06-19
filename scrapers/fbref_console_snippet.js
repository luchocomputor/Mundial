// ─────────────────────────────────────────────────────────────────────────────
// SNIPPET CONSOLE FBref → CSV  (FBref est bloqué côté serveur : on scrape DANS le
// navigateur de Louis, où il marche).
//
// USAGE :
//   1. Ouvrir une page de stats joueurs FBref, p.ex. la « Standard Stats » d'une
//      ligue (Saudi Pro League, Liga MX, Brésil, Eredivisie, Championship…).
//      → page type : fbref.com/en/comps/<id>/stats/<Ligue>-Stats
//   2. F12 → onglet Console → coller TOUT ce qui est sous la ligne, Entrée.
//   3. Un CSV se télécharge. Recommencé pour les tables Shooting / Goal-Shot
//      Creation / Possession / Passing si on veut + de profondeur (la Standard
//      suffit pour npxG + xA).
//   4. Déposer les CSV dans data/raw/fbref/raw/ puis :
//        python -m scrapers.build_wc2026_fbref
//
// Robuste : extrait par attribut data-stat (identifiants FBref stables), pas par
// en-tête visible. Le builder reconnaît ces colonnes (npxg, xg_assist, shots,
// shots_on_target, sca, touches_att_pen_area, take_ons, progressive_*, minutes_90).
// ─────────────────────────────────────────────────────────────────────────────
(() => {
  // table de stats la plus fournie de la page (évite les mini-tables annexes)
  const tables = [...document.querySelectorAll('table.stats_table')];
  if (!tables.length) { alert('Aucune table FBref (table.stats_table) sur cette page.'); return; }
  const tbl = tables.sort((a, b) => b.querySelectorAll('tbody tr').length
                                   - a.querySelectorAll('tbody tr').length)[0];
  const rows = [...tbl.querySelectorAll('tbody tr:not(.thead)')];
  const keys = new Set();
  rows.forEach(tr => tr.querySelectorAll('[data-stat]').forEach(
    c => keys.add(c.getAttribute('data-stat'))));
  const cols = [...keys];
  const esc = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [cols.join(',')];
  let n = 0;
  rows.forEach(tr => {
    const cells = {};
    tr.querySelectorAll('[data-stat]').forEach(
      c => cells[c.getAttribute('data-stat')] = c.textContent.trim());
    if (!cells.player) return;                 // saute les lignes d'espacement
    lines.push(cols.map(k => esc(cells[k])).join(','));
    n++;
  });
  const name = (document.title.split('|')[0].trim().replace(/\s+/g, '_') || 'fbref') + '.csv';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }));
  a.download = name;
  a.click();
  console.log(`FBref → ${name} : ${n} joueurs, ${cols.length} colonnes.`);
})();
