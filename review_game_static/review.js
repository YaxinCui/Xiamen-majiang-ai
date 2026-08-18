const $ = (selector) => document.querySelector(selector);
let current = null;
let selectedIndex = null;
let pendingNext = null;
let sending = false;

const NUMBERS = ['一', '二', '三', '四', '五', '六', '七', '八', '九'];
const SEAT_NAMES = ['你', '下家', '对家', '上家'];
const ACTION_NAMES = { draw: '摸牌', discard: '弃牌', chi: '吃', pong: '碰', ming_kan: '明杠', an_kan: '暗杠', add_kan: '补杠', hu: '胡' };

function tileName(id) {
  if (id < 9) return `${id + 1}万`;
  if (id < 18) return `${id - 8}筒`;
  if (id < 27) return `${id - 17}条`;
  if (id < 31) return ['东', '南', '西', '北'][id - 27];
  return ['中', '发', '白'][id - 31] || `牌${id}`;
}

function tileFace(id) {
  if (id < 9) return `${NUMBERS[id]}<br>萬`;
  if (id < 18) return `${id - 8}<br>筒`;
  if (id < 27) return `${id - 17}<br>条`;
  return tileName(id);
}

function describeAction(action) {
  if (!action) return '未知动作';
  if (action.kind === 'discard') return `打 ${tileName(action.tile)}`;
  if (action.kind === 'pass') return '过';
  const verb = ACTION_NAMES[action.kind] || action.kind;
  const target = action.tile === undefined ? '' : ` ${tileName(action.tile)}`;
  const consumed = action.tiles?.length
    ? `（使用 ${action.tiles.map(tileName).join('、')}）`
    : '';
  return `${verb}${target}${consumed}`;
}

function tileElement(id, { compact = false, clickable = false, actionIndex = null, drawn = false } = {}) {
  const node = document.createElement('button');
  node.type = 'button';
  node.className = 'tile';
  if (compact) node.classList.add('compact');
  if (clickable) node.classList.add('clickable');
  if (drawn) node.classList.add('drawn');
  if (id < 9) node.classList.add('wan');
  else if (id < 18) node.classList.add('tong');
  else if (id < 27) node.classList.add('tiao');
  else if (id < 31) node.classList.add('honor');
  else node.classList.add('dragon');
  if (current?.state?.gold_tile === id) node.classList.add('gold');
  if (selectedIndex === actionIndex && actionIndex !== null) node.classList.add('selected');
  node.innerHTML = tileFace(id);
  node.title = tileName(id);
  node.setAttribute('aria-label', tileName(id));
  node.disabled = !clickable || sending || pendingNext !== null;
  if (clickable) node.addEventListener('click', () => selectAction(actionIndex));
  return node;
}

function renderTiles(container, tiles, options = {}) {
  container.replaceChildren();
  tiles.forEach((tile) => container.append(tileElement(tile, options)));
}

function playerCard(player, actor = false) {
  const card = document.createElement('article');
  card.className = `player-card${actor ? ' actor-card' : ''}`;
  const title = document.createElement('div');
  title.className = 'player-title';
  title.innerHTML = `<span>${SEAT_NAMES[player.relative_seat]}${current.state.dealer_relative === player.relative_seat ? ' · 庄' : ''}</span><span class="seat-meta">暗牌 ${player.hand_count} · 花 ${player.flowers}</span>`;
  card.append(title);

  if (player.melds.length) {
    const label = document.createElement('p'); label.className = 'row-label'; label.textContent = '副露'; card.append(label);
    const meldRow = document.createElement('div'); meldRow.className = 'tile-row';
    player.melds.forEach((meld) => {
      const group = document.createElement('span'); group.className = 'meld'; group.title = ACTION_NAMES[meld.kind] || meld.kind;
      meld.tiles.forEach((tile) => group.append(tileElement(tile, { compact: true })));
      meldRow.append(group);
    });
    card.append(meldRow);
  }
  const riverLabel = document.createElement('p'); riverLabel.className = 'row-label'; riverLabel.textContent = '弃牌河'; card.append(riverLabel);
  const river = document.createElement('div'); river.className = 'tile-row';
  player.discards.forEach((tile) => river.append(tileElement(tile, { compact: true })));
  if (!player.discards.length) river.textContent = '—';
  card.append(river);
  return card;
}

function actorCard() {
  const actor = current.state.public_players.find((player) => player.relative_seat === 0);
  const card = playerCard(actor, true);
  card.id = 'actor-seat';
  const discardMode = current.legal_actions.every((action) => action.kind === 'discard');
  const label = document.createElement('p'); label.className = 'row-label'; label.textContent = discardMode ? '你的手牌（点击选择弃牌）' : '你的手牌（响应判断）'; card.append(label);
  const hand = document.createElement('div'); hand.className = 'tile-row';
  const actionsByTile = new Map(current.legal_actions.flatMap((action, index) => action.kind === 'discard' ? [[action.tile, index]] : []));
  const tiles = [...current.state.hand];
  let drawn = null;
  if (current.state.drawn_tile !== null && current.state.drawn_tile !== undefined) {
    const at = tiles.lastIndexOf(current.state.drawn_tile);
    if (at >= 0) [drawn] = tiles.splice(at, 1);
  }
  tiles.forEach((tile) => hand.append(tileElement(tile, { clickable: actionsByTile.has(tile), actionIndex: actionsByTile.get(tile) })));
  if (drawn !== null) hand.append(tileElement(drawn, { clickable: actionsByTile.has(drawn), actionIndex: actionsByTile.get(drawn), drawn: true }));
  card.append(hand);
  return card;
}

function renderActionOptions() {
  const container = $('#action-options');
  container.replaceChildren();
  const responseMode = current.legal_actions.some((action) => action.kind !== 'discard');
  container.classList.toggle('hidden', !responseMode);
  if (!responseMode) return;
  current.legal_actions.forEach((action, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'action-choice';
    button.textContent = describeAction(action);
    button.disabled = sending || pendingNext !== null;
    if (selectedIndex === index) button.classList.add('selected');
    button.addEventListener('click', () => selectAction(index));
    container.append(button);
  });
}

function renderHistory() {
  const list = $('#history-list'); list.replaceChildren();
  const rows = current.state.recent_public_actions || [];
  rows.slice(-10).reverse().forEach((action) => {
    const item = document.createElement('li');
    const seat = SEAT_NAMES[action.relative_seat] || `座位${action.relative_seat}`;
    const face = action.tile === undefined ? '' : ` ${tileName(action.tile)}`;
    item.textContent = `第${action.turn}巡 · ${seat} ${ACTION_NAMES[action.kind] || action.kind}${face}`;
    list.append(item);
  });
  if (!rows.length) list.innerHTML = '<li>尚无公开动作</li>';
}

function updateProgress(progress) {
  $('#progress-count').textContent = `${progress.completed} / ${progress.total}`;
  $('#progress-note').textContent = progress.skipped_this_session ? `本次跳过 ${progress.skipped_this_session} 题` : '已完成 / 总题数';
}

function renderState(payload) {
  current = payload; selectedIndex = null; pendingNext = null; sending = false;
  updateProgress(payload.progress);
  $('#feedback').classList.add('hidden'); $('#next-button').classList.add('hidden');
  if (payload.status !== 'reviewing') {
    $('#review-area').classList.add('hidden');
    $('#notice').classList.remove('hidden');
    $('#notice').textContent = payload.message;
    return;
  }
  $('#notice').classList.add('hidden'); $('#review-area').classList.remove('hidden');
  const state = payload.state;
  $('#gold-tile').textContent = tileName(state.gold_tile);
  $('#gold-indicator').textContent = tileName(state.gold_indicator);
  $('#wall-remaining').textContent = state.wall_remaining;
  $('#turn-count').textContent = state.turn_count;
  $('#dealer-seat').textContent = SEAT_NAMES[state.dealer_relative];
  $('#last-discard').textContent = state.last_discard === null || state.last_discard === undefined ? '—' : tileName(state.last_discard);
  $('#gold-note').textContent = state.gold_tile === 33 ? '本局白板就是真金。' : `白板按 ${tileName(state.gold_tile)} 使用。`;
  const opponents = $('#opponents'); opponents.replaceChildren();
  [3, 2, 1].forEach((seat) => opponents.append(playerCard(state.public_players.find((player) => player.relative_seat === seat))));
  const actor = $('#actor-seat'); actor.replaceWith(actorCard());
  renderHistory(); renderActionOptions(); updateSelection();
}

function selectAction(index) {
  if (sending || pendingNext) return;
  selectedIndex = index;
  document.querySelectorAll('.tile.selected').forEach((node) => node.classList.remove('selected'));
  document.querySelectorAll('.actor-card .tile.clickable').forEach((node) => {
    if (current.legal_actions[index].kind === 'discard' && node.title === tileName(current.legal_actions[index].tile)) node.classList.add('selected');
  });
  document.querySelectorAll('.action-choice').forEach((node, actionIndex) => node.classList.toggle('selected', actionIndex === index));
  updateSelection();
}

function updateSelection() {
  const selected = selectedIndex === null ? null : current.legal_actions[selectedIndex];
  $('#selected-answer').textContent = selected ? `选择：${describeAction(selected)}` : '尚未选择';
  $('#confirm-button').disabled = !selected || sending || pendingNext;
  $('#uncertain-button').disabled = !selected || sending || pendingNext;
  $('#skip-button').disabled = sending || pendingNext;
}

async function post(path, payload) {
  const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || '请求失败');
  return body;
}

async function submit(confidence) {
  if (selectedIndex === null || sending) return;
  sending = true; updateSelection();
  try {
    const result = await post('/api/review/label', { item_id: current.item_id, chosen_index: selectedIndex, confidence });
    pendingNext = result.next;
    updateProgress(result.next.progress);
    const feedback = $('#feedback');
    const human = describeAction(result.feedback.human_action);
    const teacher = describeAction(result.feedback.reference_teacher_action);
    const slowExpert = result.feedback.slow_expert_action
      ? describeAction(result.feedback.slow_expert_action)
      : null;
    feedback.innerHTML = result.feedback.agrees_with_teacher
      ? `<strong>已保存。</strong>你与规则 Teacher 都选择${human}。`
      : `<strong>已保存一条纠错分歧。</strong>你选择${human}，规则 Teacher 选择${teacher}。`;
    if (slowExpert) {
      const label = result.feedback.slow_expert_label || 'SlowExpert';
      feedback.innerHTML += `<br />${label} 选择${slowExpert}（仅在提交后显示）。`;
    }
    feedback.classList.toggle('disagree', !result.feedback.agrees_with_teacher);
    feedback.classList.remove('hidden'); $('#next-button').classList.remove('hidden');
  } catch (error) {
    sending = false; updateSelection(); showError(error.message);
  }
}

async function skip() {
  if (sending || !current) return;
  sending = true; updateSelection();
  try { renderState(await post('/api/review/skip', { item_id: current.item_id })); }
  catch (error) { sending = false; updateSelection(); showError(error.message); }
}

function showError(message) {
  const notice = $('#notice'); notice.textContent = message; notice.classList.add('error'); notice.classList.remove('hidden');
}

$('#confirm-button').addEventListener('click', () => submit('confirmed'));
$('#uncertain-button').addEventListener('click', () => submit('uncertain'));
$('#skip-button').addEventListener('click', skip);
$('#next-button').addEventListener('click', () => renderState(pendingNext));

document.addEventListener('keydown', (event) => {
  if (sending || !current) return;
  if (event.key === 'Enter') {
    event.preventDefault();
    if (pendingNext) renderState(pendingNext);
    else if (selectedIndex !== null) submit('confirmed');
    return;
  }
  if (pendingNext) return;
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    const count = current.legal_actions.length;
    const direction = event.key === 'ArrowRight' ? 1 : -1;
    const next = selectedIndex === null
      ? (direction > 0 ? 0 : count - 1)
      : (selectedIndex + direction + count) % count;
    selectAction(next);
  } else if (event.key.toLowerCase() === 'u' && selectedIndex !== null) {
    event.preventDefault(); submit('uncertain');
  } else if (event.key.toLowerCase() === 's') {
    event.preventDefault(); skip();
  }
});

fetch('/api/review').then(async (response) => {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || '载入失败');
  renderState(payload);
}).catch((error) => showError(error.message));
