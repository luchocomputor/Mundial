// ─────────────────────────────────────────────────────────────────────────────
// SNIPPET CONSOLE SOFASCORE → CSV  (xG/xA MONDIAL, Saudi/Liga MX inclus)
//
// L'app Sofascore appelle son API en SAME-ORIGIN (www.sofascore.com/api/v1/…) avec
// un en-tête anti-bot X-Requested-With. On reproduit exactement ça depuis ta session
// (URL relative = same-origin = cookies envoyés tout seuls, pas de mur CORS).
//
// USAGE :
//   1. Ouvre la page Sofascore de la ligue (URL finissant par l'id du tournoi,
//      ex .../saudi-pro-league/955). Reste sur www.sofascore.com.
//   2. F12 → Console → colle TOUT → Entrée. Un CSV se télécharge.
//   3. Dépose-le dans data/raw/fbref/raw/ → python -m scrapers.build_wc2026_fbref
//
//   La console logge les CHAMPS dispos sur le 1er joueur : si « expectedGoals »
//   n'y est pas, dis-le-moi (je changerai le paramètre group/fields).
//
// ⚠️ X-Requested-With ('416bcd') vient de TA session. S'il périme (403), refais un
//    « Copy as cURL » d'une requête statistics et donne-moi la nouvelle valeur.
// ─────────────────────────────────────────────────────────────────────────────
(async () => {
  const ut = (location.pathname.match(/\/(\d+)\/?(?:[#?].*)?$/) || [])[1];
  if (!ut) { alert("URL sans id de tournoi (doit finir par un nombre, ex .../955)."); return; }
  const XRW = '416bcd';   // en-tête anti-bot capté via Copy-as-cURL (ta session)
  const j = async path => {
    const r = await fetch(path, { headers: { Accept: '*/*', 'X-Requested-With': XRW } });
    if (!r.ok) throw new Error(`${r.status} sur ${path}`);
    return r.json();
  };
  // group=summary ne renvoie PAS minutesPlayed/xA/tirs → on demande des champs EXPLICITES.
  const fields = ['goals', 'assists', 'expectedGoals', 'expectedAssists', 'totalShots',
                  'shotsOnTarget', 'keyPasses', 'successfulDribbles', 'minutesPlayed',
                  'appearances'].join('%2C');
  const fetchSeason = async sid => {
    let r = [], page = 0, pages = 1;
    do {
      const d = await j(`/api/v1/unique-tournament/${ut}/season/${sid}/statistics`
        + `?limit=100&offset=${page * 100}&accumulation=total&fields=${fields}&order=-expectedGoals`);
      (d.results || []).forEach(x => r.push(x));
      pages = d.pages || 1; page++;
    } while (page < pages && page < 30);
    return r;
  };
  // saison : celle du HASH d'URL (#id:77012) si présente ; sinon on essaie les 4 saisons
  // les plus récentes et on garde la 1re qui renvoie des données (évite la saison vide
  // 2026/27 déjà créée, ou les phases de playoff Belgique/Danemark sans agrégat).
  const hashSeason = (location.hash.match(/id:(\d+)/) || [])[1];
  const candidates = hashSeason ? [hashSeason]
    : (await j(`/api/v1/unique-tournament/${ut}/seasons`)).seasons.slice(0, 4).map(s => s.id);
  let rows = [], season = null;
  for (const sid of candidates) {
    const r = await fetchSeason(sid);
    if (r.length) { rows = r; season = sid; break; }
  }
  if (!rows.length) { alert(`Aucune saison avec données (essayé ${candidates.join(', ')}). `
    + `Ouvre l'onglet Stats de la ligue (l'URL doit finir par #id:...,tab:stats).`); return; }
  console.log('Champs dispos sur 1 joueur :', Object.keys(rows[0]).filter(k => k !== 'player' && k !== 'team'));
  const cols = ['player_name', 'team_title', 'minutesPlayed', 'goals', 'assists', 'totalShots',
                'shotsOnTarget', 'keyPasses', 'successfulDribbles', 'expectedGoals', 'expectedAssists'];
  const get = (r, k) => k === 'player_name' ? (r.player && r.player.name)
                      : k === 'team_title' ? (r.team && r.team.name) : r[k];
  const esc = v => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const csv = [cols.join(',')].concat(rows.map(r => cols.map(k => esc(get(r, k))).join(','))).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = `sofascore_${ut}_${season}.csv`;
  a.click();
  console.log(`Sofascore → ${rows.length} joueurs, tournoi ${ut} saison ${season}.`);
})().catch(e => alert('Erreur Sofascore : ' + (e && e.message ? e.message : e)
  + '\n(403 = token X-Requested-With périmé → refais un Copy-as-cURL et redonne la valeur)'));
