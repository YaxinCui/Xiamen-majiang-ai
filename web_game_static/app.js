const $ = (selector) => document.querySelector(selector);
const API_BASE = document.body.dataset.apiBase || '/api/game';
const FIXED_RULES_PROFILE = document.body.dataset.rulesProfile || null;
let state = null;
let sending = false;
let debugAiHands = true;
let selectedRulesProfile = null;

const MELD_LABELS = { chi: '吃', pong: '碰', ming_kan: '明杠', an_kan: '暗杠', add_kan: '补杠' };
const CHINESE_NUMERALS = ['一', '二', '三', '四', '伍', '六', '七', '八', '九'];
const PIP_POSITIONS = {
  1: [5], 2: [3, 7], 3: [3, 5, 7], 4: [1, 3, 7, 9], 5: [1, 3, 5, 7, 9],
  6: [1, 3, 4, 6, 7, 9], 7: [1, 3, 4, 5, 6, 7, 9], 8: [1, 2, 3, 4, 6, 7, 8, 9],
  9: [1, 2, 3, 4, 5, 6, 7, 8, 9],
};

function tileFace(tile) {
  const face = document.createElement('span');
  face.className = 'tile-face';
  const id = tile.id;

  if (id < 9) {
    face.classList.add('tile-wan');
    face.innerHTML = `<span class="tile-rank">${CHINESE_NUMERALS[id]}</span><span class="tile-unit">萬</span>`;
    return face;
  }
  if (id < 27) {
    const isTong = id < 18;
    const rank = id % 9 + 1;
    face.classList.add(isTong ? 'tile-tong' : 'tile-tiao', `rank-${rank}`);
    if (isTong && rank === 1) {
      face.classList.add('tile-one-circle');
      face.innerHTML = '<svg class="one-circle-wheel" viewBox="0 0 48 58" aria-hidden="true"><circle cx="24" cy="29" r="20" fill="#f5ead6" stroke="#226f92" stroke-width="2.4"/><circle cx="24" cy="29" r="16.5" fill="none" stroke="#2f8a5b" stroke-width="2"/><g stroke="#165b79" stroke-width=".8"><ellipse cx="24" cy="17" rx="3.6" ry="7.2" fill="#2a769c"/><ellipse cx="24" cy="17" rx="3.6" ry="7.2" fill="#3b955f" transform="rotate(45 24 29)"/><ellipse cx="24" cy="17" rx="3.6" ry="7.2" fill="#2a769c" transform="rotate(90 24 29)"/><ellipse cx="24" cy="17" rx="3.6" ry="7.2" fill="#3b955f" transform="rotate(135 24 29)"/><ellipse cx="24" cy="17" rx="3.6" ry="7.2" fill="#2a769c" transform="rotate(180 24 29)"/><ellipse cx="24" cy="17" rx="3.6" ry="7.2" fill="#3b955f" transform="rotate(225 24 29)"/><ellipse cx="24" cy="17" rx="3.6" ry="7.2" fill="#2a769c" transform="rotate(270 24 29)"/><ellipse cx="24" cy="17" rx="3.6" ry="7.2" fill="#3b955f" transform="rotate(315 24 29)"/></g><circle cx="24" cy="29" r="8" fill="#f7eddc" stroke="#bb3f35" stroke-width="2"/><path d="M24 20.8l2.2 5.5 5.9-1.4-3.7 4.8 3.7 4.7-5.9-1.3-2.2 5.5-2.2-5.5-5.9 1.3 3.7-4.7-3.7-4.8 5.9 1.4z" fill="#c64638"/><circle cx="24" cy="29" r="2.3" fill="#f2d45d" stroke="#286b8b" stroke-width="1"/></svg>';
      return face;
    }
    if (!isTong && rank === 1) {
      face.classList.add('tile-one-bamboo');
      face.innerHTML = `
        <svg class="bamboo-bird" viewBox="0 0 44 58" aria-hidden="true">
          <path d="M14 47c2-12 8-20 17-29-1 12-4 24-11 33z" fill="#19764f"/>
          <path d="M19 45c-1-12 2-24 8-36 4 14 3 27-2 39z" fill="#287baf"/>
          <path d="M24 45c2-10 8-20 14-27-1 13-4 24-10 31z" fill="#c64638"/>
          <ellipse cx="19" cy="24" rx="8" ry="11" fill="#2f8b56"/>
          <path d="M14 22c6 1 11 4 14 9-6 1-11-1-15-5z" fill="#176b9c"/>
          <circle cx="18" cy="12" r="6" fill="#308d59"/>
          <path d="M22 11l8 3-8 3z" fill="#d84b38"/>
          <circle cx="20" cy="11" r="1.2" fill="#102f29"/>
          <path d="M15 7l-2-4m5 4V2m3 6 3-4" stroke="#277b55" stroke-width="1.8" stroke-linecap="round"/>
        </svg>`;
      return face;
    }
    const grid = document.createElement('span');
    grid.className = 'pip-grid';
    PIP_POSITIONS[rank].forEach((position, index) => {
      const pip = document.createElement('i');
      pip.className = `pip pip-${position} ${pipColorClass(rank, position, index)}`;
      grid.append(pip);
    });
    face.append(grid);
    return face;
  }
  if (id < 31) {
    face.classList.add('tile-wind');
    face.textContent = ['东', '南', '西', '北'][id - 27];
    return face;
  }
  if (id < 34) {
    face.classList.add('tile-dragon', `dragon-${id - 31}`);
    if (id === 33) {
      face.classList.add('white-dragon');
      face.innerHTML = '<span class="white-dragon-frame" aria-hidden="true"></span>';
    } else {
      face.textContent = ['中', '發'][id - 31];
    }
    return face;
  }
  face.classList.add('tile-flower');
  face.innerHTML = `<span>❀</span><small>${tile.name}</small>`;
  return face;
}

function pipColorClass(rank, position, index) {
  if (rank === 1) return 'pip-multicolor';
  if (rank === 2) return index === 0 ? 'pip-green' : 'pip-blue';
  if (rank === 3) return ['pip-green', 'pip-red', 'pip-blue'][index];
  if (rank === 4) return [1, 9].includes(position) ? 'pip-green' : 'pip-blue';
  if (rank === 5) return position === 5 ? 'pip-red' : ([1, 9].includes(position) ? 'pip-green' : 'pip-blue');
  if (rank === 6) return position <= 3 ? 'pip-green' : 'pip-red';
  if (rank === 7) return [1, 5, 9].includes(position) ? 'pip-green' : 'pip-red';
  if (rank === 8) return 'pip-blue';
  if (rank === 9) return position <= 3 ? 'pip-green' : (position <= 6 ? 'pip-red' : 'pip-blue');
  return 'pip-blue';
}

function tileElement(tile, {
  clickable = false,
  action = null,
  compact = false,
  forced = false,
  drawn = false,
  lastDiscard = false,
} = {}) {
  const node = document.querySelector('#tile-template').content.firstElementChild.cloneNode(true);
  node.dataset.tile = tile.id;
  node.setAttribute('aria-label', tile.name);
  node.title = tile.name;
  const corner = document.createElement('span');
  corner.className = 'tile-corner';
  corner.textContent = tile.name;
  node.append(corner, tileFace(tile));
  node.classList.toggle('gold', state?.gold_tile?.id === tile.id);
  node.classList.toggle('gold-proxy', Boolean(state?.rules?.white_dragon_is_gold_proxy) && tile.id === 33);
  node.classList.toggle('compact', compact);
  node.classList.toggle('forced', forced);
  node.classList.toggle('drawn-tile', drawn);
  node.classList.toggle('last-discard', lastDiscard);
  if (drawn) node.setAttribute('aria-label', `刚摸到的 ${tile.name}`);
  if (!clickable) {
    node.disabled = true;
    node.classList.add('display-tile');
  } else {
    node.title = action?.label || `打出 ${tile.name}`;
    node.addEventListener('click', () => sendAction(action || { kind: 'discard', tile: tile.id }));
  }
  return node;
}

function renderTileSlot(selector, tile) {
  const slot = $(selector);
  slot.replaceChildren();
  if (tile) slot.append(tileElement(tile, { compact: true }));
  else slot.textContent = '–';
}

function renderSeat(player) {
  const seat = document.querySelector(`[data-seat="${player.seat}"]`);
  seat.replaceChildren();
  seat.classList.toggle('active', state.phase === 'discard' && state.current_player === player.seat);
  const header = document.createElement('div');
  header.className = 'seat-header';
  header.innerHTML = `<strong>${player.name}${state.dealer === player.seat ? ' · 庄' : ''}</strong><span>${player.score > 0 ? '+' : ''}${player.score} 分</span>`;
  seat.append(header);

  const meta = document.createElement('p');
  meta.className = 'seat-meta';
  meta.textContent = player.seat === 0
    ? `手牌 ${player.hand_count} 张`
    : `${debugAiHands ? '调试手牌' : '暗牌'} ${player.hand_count} 张`;
  if (player.flowers.length) meta.textContent += ` · 花 ${player.flowers.map((tile) => tile.name).join(' ')}`;
  if (player.status?.length) meta.textContent += ` · ${player.status.join(' · ')}`;
  seat.append(meta);

  const melds = document.createElement('div');
  melds.className = 'meld-row';
  player.melds.forEach((meld) => {
    const group = document.createElement('span');
    group.className = `meld meld-${meld.kind}`;
    group.title = MELD_LABELS[meld.kind] || meld.kind;
    meld.tiles.forEach((tile) => group.append(tileElement(tile, { compact: true })));
    melds.append(group);
  });
  if (melds.children.length) seat.append(melds);

  if ((player.seat === 0 || debugAiHands) && player.hand) {
    const hand = document.createElement('div');
    hand.className = player.seat === 0 ? 'hand-row' : 'hand-row ai-hand-row';
    const discardActions = new Map(state.actions
      .filter((action) => action.kind === 'discard')
      .map((action) => [action.tile, action]));
    const forcedTiles = new Set((state.forced_discards || []).map((tile) => tile.id));
    const handTiles = [...player.hand];
    let drawnTile = null;
    if (player.drawn_tile) {
      const drawnIndex = handTiles.map((tile) => tile.id).lastIndexOf(player.drawn_tile.id);
      if (drawnIndex >= 0) [drawnTile] = handTiles.splice(drawnIndex, 1);
    }
    const appendHandTile = (tile, drawn = false) => hand.append(tileElement(tile, {
      clickable: player.seat === 0 && discardActions.has(tile.id),
      action: player.seat === 0 ? discardActions.get(tile.id) : null,
      compact: player.seat !== 0,
      forced: forcedTiles.has(tile.id),
      drawn,
    }));
    handTiles.forEach((tile) => appendHandTile(tile));
    if (drawnTile) appendHandTile(drawnTile, true);
    seat.append(hand);
  } else {
    const backs = document.createElement('div');
    backs.className = 'back-row';
    for (let index = 0; index < Math.min(player.hand_count, 17); index += 1) {
      const back = document.createElement('span');
      back.className = 'tile tile-back compact';
      backs.append(back);
    }
    seat.append(backs);
  }
}

function renderRiver(player) {
  const river = document.querySelector(`[data-river="${player.seat}"]`);
  river.replaceChildren();
  const discardLimit = [1, 3].includes(player.seat) ? 12 : 18;
  const hiddenDiscardCount = Math.max(0, player.discards.length - discardLimit);
  if (hiddenDiscardCount) {
    const overflow = document.createElement('span');
    overflow.className = 'discard-overflow';
    overflow.textContent = `+${hiddenDiscardCount}`;
    overflow.title = `另有 ${hiddenDiscardCount} 张更早的弃牌`;
    river.append(overflow);
  }
  player.discards.slice(-discardLimit).forEach((tile, index, visibleDiscards) => {
    const isLatest = state.latest_discard_seat === player.seat
      && index === visibleDiscards.length - 1
      && state.latest_discard?.id === tile.id;
    river.append(tileElement(tile, { compact: true, lastDiscard: isLatest }));
  });
}

function renderActions() {
  const panel = $('#action-panel');
  panel.replaceChildren();
  const nonDiscard = state.actions.filter((action) => action.kind !== 'discard');
  if (state.phase === 'over') {
    panel.innerHTML = '<p><strong>本局已结算</strong>，可查看结果或开始下一局。</p>';
    return;
  }
  if (!state.actions.length) {
    panel.innerHTML = '<p>AI 正在思考，或等待其他玩家响应…</p>';
    return;
  }
  const intro = document.createElement('p');
  intro.textContent = state.phase === 'response'
    ? `响应 ${state.last_discard?.name || '上一张弃牌'}：`
    : '轮到你：点击亮起的手牌出牌';
  panel.append(intro);
  if (state.forced_discards?.length) {
    const notice = document.createElement('span');
    notice.className = 'rule-notice';
    notice.textContent = `跟打：请先打 ${state.forced_discards.map((tile) => tile.name).join('、')}`;
    panel.append(notice);
  }
  nonDiscard.forEach((action) => {
    const button = document.createElement('button');
    button.className = ['hu', 'advance_tour'].includes(action.kind) ? 'primary-button' : 'action-button';
    button.textContent = action.label;
    button.addEventListener('click', () => sendAction(action));
    panel.append(button);
  });
}

function renderResult() {
  const result = $('#result-card');
  if (state.phase !== 'over') {
    result.classList.add('hidden');
    result.replaceChildren();
    return;
  }
  result.classList.remove('hidden');
  const winLabels = {
    discard: '点炮胡', self_draw: '自摸', travelling_gold: '游金',
    double_travelling: '双游', triple_travelling: '三游',
    three_gold_open: '开局三金倒', three_gold: '三金倒',
    opening_wait: '天听自摸', opening_gold: '抢金', heaven: '天胡',
  };
  const title = state.winner === null
    ? '本局流局'
    : `${state.players[state.winner].name} ${winLabels[state.win_type] || '胡牌'}`;
  const content = document.createElement('div');
  content.innerHTML = `<p class="section-kicker">本局结算</p><h2>${title}</h2><p>${state.win_pattern || '牌墙耗尽'} · ${state.message}</p>`;
  const breakdown = state.score_breakdown;
  if (breakdown) {
    const score = document.createElement('p');
    score.className = 'score-line';
    const payer = breakdown.payment_mode === 'all_pay' ? '其余三家各付' : '放铳者付';
    score.textContent = breakdown.mode === 'new120_fixed'
      ? `主分 ${breakdown.base} + 水 ${breakdown.water} = ${breakdown.unit}；${payer} ${breakdown.per_payer}`
      : `底 ${breakdown.base} + 水 ${breakdown.water} = ${breakdown.unit}；倍数 ×${breakdown.multiplier}；${payer} ${breakdown.per_payer}`;
    content.append(score);
    if (breakdown.items?.length) {
      const details = document.createElement('p');
      details.className = 'score-detail';
      details.textContent = breakdown.items.map((item) => `${item.label} +${item.water}水`).join(' · ');
      content.append(details);
    }
  }
  result.append(content);
  const button = document.createElement('button');
  button.className = 'primary-button';
  button.textContent = '再来一局';
  button.addEventListener('click', () => newGame(false));
  result.append(button);
}

function render() {
  if (!state) return;
  $('#message').textContent = state.message;
  $('#turn-detail').textContent = state.phase === 'over' ? '本局已结束' : `当前：${state.players[state.current_player].name}`;
  $('#wall-count').textContent = state.wall_remaining;
  $('#turn-count').textContent = state.turn_count;
  $('#hand-number').textContent = state.hand_number || 1;
  $('#dealer-streak').textContent = state.dealer_streak ? `${state.dealer_streak} 连` : '首庄';
  $('#gold-indicator-label').textContent = state.gold_indicator_label || '指示牌';
  renderTileSlot('#gold-indicator', state.gold_indicator);
  renderTileSlot('#gold-tile', state.gold_tile);
  $('#gold-indicator').title = state.gold_dice ? `翻金骰子：${state.gold_dice.join(' + ')}` : '';
  const goldNote = $('#gold-proxy-note');
  if (state.rules.white_dragon_proxy_enabled && state.gold_tile?.id === 33) {
    goldNote.textContent = '本局白板就是真金。';
  } else if (state.rules.white_dragon_is_gold_proxy && state.gold_tile) {
    goldNote.textContent = `白板按 ${state.gold_tile.name} 使用，不是万能牌。`;
  } else {
    goldNote.textContent = '';
  }
  $('#phase-chip').textContent = ({ discard: '出牌', response: '响应', over: '结算' })[state.phase] || state.phase;
  const special = $('#special-state');
  if (state.tour_state) {
    const label = ({ 1: '游金', 2: '双游', 3: '三游' })[state.tour_state.level];
    special.textContent = `${state.players[state.tour_state.owner].name} · ${label}`;
    special.classList.remove('hidden');
  } else if (state.opening_wait_seats?.length) {
    special.textContent = `天听 ${state.opening_wait_seats.length} 家`;
    special.classList.remove('hidden');
  } else {
    special.classList.add('hidden');
  }
  $('#rules-list').replaceChildren(...state.rules.summary.map((rule) => {
    const item = document.createElement('li'); item.textContent = rule; return item;
  }));
  state.players.forEach((player) => {
    renderSeat(player);
    renderRiver(player);
  });
  const events = $('#event-list');
  events.replaceChildren(...state.events.map((event) => {
    const item = document.createElement('li');
    item.innerHTML = `<span>${event.kind}</span>${event.text}`;
    return item;
  }));
  renderActions();
  renderResult();
  renderProfilePicker();
  renderDebugToggle();
  $('#new-game').textContent = state.phase === 'over' ? '下一局' : '新开一局';
}

function renderProfilePicker() {
  const picker = $('#rules-profile');
  if (!picker || !state?.rule_profiles) return;
  if (!selectedRulesProfile) selectedRulesProfile = state.rules.profile;
  const profiles = FIXED_RULES_PROFILE
    ? state.rule_profiles.filter((profile) => profile.id === FIXED_RULES_PROFILE)
    : state.rule_profiles.filter((profile) => profile.id !== 'new120');
  picker.replaceChildren(...profiles.map((profile) => {
    const option = document.createElement('option');
    option.value = profile.id;
    option.textContent = profile.name;
    option.title = profile.description;
    return option;
  }));
  picker.value = selectedRulesProfile;
}

function renderDebugToggle() {
  const toggle = $('#toggle-debug');
  toggle.textContent = debugAiHands ? '隐藏 AI 手牌（调试）' : '显示 AI 手牌（调试）';
  toggle.setAttribute('aria-pressed', String(debugAiHands));
  toggle.classList.toggle('is-active', debugAiHands);
  $('#debug-ribbon').classList.toggle('hidden', !debugAiHands);
}

async function request(path, body = null) {
  const response = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}

async function sendAction(action) {
  if (sending) return;
  sending = true;
  try {
    state = await request(`${API_BASE}/action`, action);
    if (debugAiHands) state = await requestGameState();
    render();
  } catch (error) {
    window.alert(error.message);
  } finally {
    sending = false;
  }
}

async function newGame(resetMatch = false) {
  if (sending) return;
  if (state) {
    const question = resetMatch
      ? (state.phase === 'over' ? '清空当前积分并重新开始？' : '重置积分并放弃当前牌局？')
      : (state.phase === 'over' ? null : '放弃当前牌局并新开一局？');
    if (question && !window.confirm(question)) return;
  }
  sending = true;
  try {
    state = await request(`${API_BASE}/new`, {
      rules_profile: FIXED_RULES_PROFILE || selectedRulesProfile || $('#rules-profile')?.value,
      reset_match: resetMatch,
    });
    selectedRulesProfile = state.rules.profile;
    if (debugAiHands) state = await requestGameState();
    render();
  } finally {
    sending = false;
  }
}

async function requestGameState() {
  return request(`${API_BASE}${debugAiHands ? '?debug=1' : ''}`);
}

async function toggleDebugAiHands() {
  if (sending) return;
  sending = true;
  try {
    debugAiHands = !debugAiHands;
    if (debugAiHands) state = await requestGameState();
    render();
  } catch (error) {
    debugAiHands = false;
    window.alert(error.message);
  } finally {
    sending = false;
  }
}

$('#new-game').addEventListener('click', () => newGame(false));
$('#reset-match').addEventListener('click', () => newGame(true));
$('#toggle-debug').addEventListener('click', toggleDebugAiHands);
$('#rules-profile')?.addEventListener('change', (event) => {
  selectedRulesProfile = event.target.value;
});
requestGameState().then((data) => { state = data; selectedRulesProfile = data.rules.profile; render(); }).catch((error) => {
  $('#message').textContent = `无法连接服务：${error.message}`;
});
