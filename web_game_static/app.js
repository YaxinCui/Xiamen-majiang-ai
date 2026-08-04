const $ = (selector) => document.querySelector(selector);
let state = null;
let sending = false;

function tileElement(tile, { clickable = false, action = null, compact = false } = {}) {
  const node = document.querySelector('#tile-template').content.firstElementChild.cloneNode(true);
  node.textContent = tile.name;
  node.dataset.tile = tile.id;
  node.classList.toggle('gold', state?.gold_tile?.id === tile.id);
  node.classList.toggle('compact', compact);
  if (!clickable) {
    node.disabled = true;
    node.classList.add('display-tile');
  } else {
    node.title = `打出 ${tile.name}`;
    node.addEventListener('click', () => sendAction(action || { kind: 'discard', tile: tile.id }));
  }
  return node;
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
  meta.textContent = player.seat === 0 ? `手牌 ${player.hand_count} 张` : `暗牌 ${player.hand_count} 张`;
  if (player.flowers.length) meta.textContent += ` · 花 ${player.flowers.map((tile) => tile.name).join(' ')}`;
  seat.append(meta);

  const melds = document.createElement('div');
  melds.className = 'meld-row';
  player.melds.forEach((meld) => {
    const group = document.createElement('span');
    group.className = `meld meld-${meld.kind}`;
    group.title = meld.kind;
    meld.tiles.forEach((tile) => group.append(tileElement(tile, { compact: true })));
    melds.append(group);
  });
  if (melds.children.length) seat.append(melds);

  const discards = document.createElement('div');
  discards.className = 'discard-row';
  player.discards.forEach((tile) => discards.append(tileElement(tile, { compact: true })));
  seat.append(discards);

  if (player.seat === 0 && player.hand) {
    const hand = document.createElement('div');
    hand.className = 'hand-row';
    const canDiscard = state.actions.some((action) => action.kind === 'discard');
    player.hand.forEach((tile) => hand.append(tileElement(tile, { clickable: canDiscard, compact: false })));
    seat.append(hand);
  } else {
    const backs = document.createElement('div');
    backs.className = 'back-row';
    for (let index = 0; index < Math.min(player.hand_count, 14); index += 1) {
      const back = document.createElement('span');
      back.className = 'tile tile-back compact';
      backs.append(back);
    }
    seat.append(backs);
  }
}

function renderActions() {
  const panel = $('#action-panel');
  panel.replaceChildren();
  const nonDiscard = state.actions.filter((action) => action.kind !== 'discard');
  if (state.phase === 'over') return;
  if (!state.actions.length) {
    panel.innerHTML = '<p>AI 正在思考，或等待其他玩家响应…</p>';
    return;
  }
  const intro = document.createElement('p');
  intro.textContent = state.phase === 'response' ? '响应上一张弃牌：' : '点击手牌出牌，或选择特殊动作：';
  panel.append(intro);
  nonDiscard.forEach((action) => {
    const button = document.createElement('button');
    button.className = action.kind === 'hu' ? 'primary-button' : 'action-button';
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
  const title = state.winner === null ? '本局流局' : `${state.players[state.winner].name} ${state.win_type === 'self_draw' ? '自摸' : '胡牌'}`;
  result.innerHTML = `<div><p class="section-kicker">本局结算</p><h2>${title}</h2><p>${state.win_pattern || '牌墙耗尽'} · ${state.message}</p></div>`;
  const button = document.createElement('button');
  button.className = 'primary-button';
  button.textContent = '再来一局';
  button.addEventListener('click', newGame);
  result.append(button);
}

function render() {
  if (!state) return;
  $('#message').textContent = state.message;
  $('#turn-detail').textContent = state.phase === 'over' ? '本局已结束' : `当前：${state.players[state.current_player].name}`;
  $('#wall-count').textContent = state.wall_remaining;
  $('#turn-count').textContent = state.turn_count;
  $('#gold-indicator').textContent = state.gold_indicator?.name || '–';
  $('#gold-tile').textContent = state.gold_tile?.name || '–';
  $('#phase-chip').textContent = ({ discard: '出牌', response: '响应', over: '结算' })[state.phase] || state.phase;
  $('#rules-list').replaceChildren(...state.rules.summary.map((rule) => {
    const item = document.createElement('li'); item.textContent = rule; return item;
  }));
  state.players.forEach(renderSeat);
  const events = $('#event-list');
  events.replaceChildren(...state.events.map((event) => {
    const item = document.createElement('li');
    item.innerHTML = `<span>${event.kind}</span>${event.text}`;
    return item;
  }));
  renderActions();
  renderResult();
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
    state = await request('/api/game/action', action);
    render();
  } catch (error) {
    window.alert(error.message);
  } finally {
    sending = false;
  }
}

async function newGame() {
  if (sending) return;
  sending = true;
  try {
    state = await request('/api/game/new', {});
    render();
  } finally {
    sending = false;
  }
}

$('#new-game').addEventListener('click', newGame);
request('/api/game').then((data) => { state = data; render(); }).catch((error) => {
  $('#message').textContent = `无法连接服务：${error.message}`;
});
