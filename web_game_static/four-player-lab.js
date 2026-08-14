const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const SEAT_NAMES = ['东家', '南家', '西家', '北家'];
const MELD_LABELS = { chi: '吃', pong: '碰', ming_kan: '明杠', an_kan: '暗杠', add_kan: '补杠' };
const WIN_POINTS = { 平胡: 2, 自摸: 4, 抢金: 10, 游金: 10, 双游: 20, 三游: 80 };
const CHINESE_NUMERALS = ['一', '二', '三', '四', '伍', '六', '七', '八', '九'];
const FLOWER_NAMES = ['梅', '兰', '菊', '竹', '春', '夏', '秋', '冬'];
const PIP_POSITIONS = {
  1: [5], 2: [3, 7], 3: [3, 5, 7], 4: [1, 3, 7, 9], 5: [1, 3, 5, 7, 9],
  6: [1, 3, 4, 6, 7, 9], 7: [1, 3, 4, 5, 6, 7, 9], 8: [1, 2, 3, 4, 6, 7, 8, 9],
  9: [1, 2, 3, 4, 5, 6, 7, 8, 9],
};

let state = null;
let logSerial = 0;

function tileName(id) {
  if (id < 27) return `${id % 9 + 1}${['万', '筒', '条'][Math.floor(id / 9)]}`;
  if (id === 33) return '白';
  if (id >= 34 && id < 42) return FLOWER_NAMES[id - 34];
  return `未知牌 ${id}`;
}

function tilePayload(id) {
  return { id, name: tileName(id), flower: id >= 34 };
}

function build120Wall() {
  const wall = [];
  [...Array(27).keys(), 33].forEach((id) => {
    for (let copy = 0; copy < 4; copy += 1) wall.push(id);
  });
  for (let flower = 34; flower < 42; flower += 1) wall.push(flower);
  return wall;
}

function shuffle(tiles) {
  for (let index = tiles.length - 1; index > 0; index -= 1) {
    const target = Math.floor(Math.random() * (index + 1));
    [tiles[index], tiles[target]] = [tiles[target], tiles[index]];
  }
  return tiles;
}

function isFlower(tile) {
  return tile >= 34;
}

function isSuited(tile) {
  return tile >= 0 && tile < 27;
}

function addLog(title, detail) {
  if (!state) return;
  logSerial += 1;
  state.events.unshift({ number: logSerial, title, detail });
  state.events = state.events.slice(0, 80);
}

function drawForPlayer(seat, { replacement = false, opening = false } = {}) {
  const player = state.players[seat];
  while (state.wall.length) {
    const tile = state.wall.shift();
    if (isFlower(tile)) {
      player.flowers.push(tile);
      addLog('补花', `${SEAT_NAMES[seat]}摸到${tileName(tile)}，从牌墙继续补牌`);
      continue;
    }
    player.hand.push(tile);
    player.hand.sort((a, b) => a - b);
    player.drawnTile = opening ? null : tile;
    if (replacement) addLog('补牌', `${SEAT_NAMES[seat]}杠后补进一张牌`);
    return tile;
  }
  finishDraw();
  return null;
}

function selectGold() {
  const dice = [1 + Math.floor(Math.random() * 6), 1 + Math.floor(Math.random() * 6)];
  const start = Math.max(state.wall.length - dice[0] - dice[1], 0);
  const indices = [];
  for (let index = start; index >= 0; index -= 1) indices.push(index);
  for (let index = state.wall.length - 1; index > start; index -= 1) indices.push(index);
  const goldIndex = indices.find((index) => !isFlower(state.wall[index]));
  state.goldIndicator = state.wall.splice(goldIndex, 1)[0];
  state.goldTile = state.goldIndicator;
  state.goldDice = dice;
}

function newFourPlayerGame() {
  logSerial = 0;
  state = {
    wall: shuffle(build120Wall()),
    players: SEAT_NAMES.map((name, seat) => ({
      seat, name, hand: [], melds: [], flowers: [], discards: [], drawnTile: null,
    })),
    dealer: 0,
    currentPlayer: 0,
    phase: 'opening',
    turn: 1,
    view: 'all',
    goldIndicator: null,
    goldTile: null,
    goldDice: null,
    latestDiscard: null,
    pendingActions: [],
    turnReason: 'opening_draw',
    events: [],
  };

  for (let count = 0; count < 16; count += 1) {
    for (let seat = 0; seat < 4; seat += 1) drawForPlayer(seat, { opening: true });
  }
  selectGold();
  drawForPlayer(state.dealer);
  state.phase = 'discard';
  addLog(
    '开局',
    `东家坐庄；骰子 ${state.goldDice[0]}＋${state.goldDice[1]}，翻出${tileName(state.goldIndicator)}，本局真金就是${tileName(state.goldTile)}`,
  );
  addLog('摸牌', '东家起手 16 张，再摸一张成为 17 张；请点击一张手牌打出');
  $('#lab-result').classList.add('hidden');
  $('#lab-result').replaceChildren();
  render();
}

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
    const rank = id % 9 + 1;
    const isTong = id < 18;
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
  if (id === 33) {
    face.classList.add('tile-dragon', 'dragon-2', 'white-dragon');
    face.innerHTML = '<span class="white-dragon-frame" aria-hidden="true"></span>';
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

function tileElement(id, { clickable = false, drawn = false, lastDiscard = false } = {}) {
  const tile = tilePayload(id);
  const node = $('#lab-tile-template').content.firstElementChild.cloneNode(true);
  node.dataset.tile = id;
  node.title = tile.name;
  node.setAttribute('aria-label', tile.name);
  const corner = document.createElement('span');
  corner.className = 'tile-corner';
  corner.textContent = tile.name;
  node.append(corner, tileFace(tile));
  node.classList.toggle('gold', id === state.goldTile);
  node.classList.toggle('gold-proxy', state.goldTile !== 33 && id === 33);
  node.classList.toggle('drawn-tile', drawn);
  node.classList.toggle('last-discard', lastDiscard);
  if (clickable) {
    node.title = `打出 ${tile.name}`;
    node.addEventListener('click', () => discardTile(id));
  } else {
    node.disabled = true;
    node.classList.add('display-tile');
  }
  return node;
}

function visibleHand(seat) {
  return state.view === 'all' || Number(state.view) === seat;
}

function renderSeat(player) {
  const seat = document.querySelector(`[data-seat="${player.seat}"]`);
  seat.replaceChildren();
  seat.classList.toggle('is-turn', state.phase === 'discard' && state.currentPlayer === player.seat);
  seat.classList.toggle('is-inspecting', state.view !== 'all' && Number(state.view) === player.seat);

  const header = document.createElement('div');
  header.className = 'lab-seat-header';
  const inspect = document.createElement('button');
  inspect.type = 'button';
  inspect.textContent = `${player.name}${player.seat === state.dealer ? ' · 庄' : ''}`;
  inspect.title = `只查看${player.name}手牌`;
  inspect.addEventListener('click', () => setView(String(player.seat)));
  const count = document.createElement('span');
  count.textContent = `手牌 ${player.hand.length} · 花 ${player.flowers.length}`;
  header.append(inspect, count);
  seat.append(header);

  const meta = document.createElement('p');
  meta.className = 'lab-seat-meta';
  const goldCount = player.hand.filter((tile) => tile === state.goldTile).length;
  meta.textContent = visibleHand(player.seat)
    ? `真金 ${goldCount} 张${player.flowers.length ? ` · 花牌 ${player.flowers.map(tileName).join(' ')}` : ''}`
    : '手牌已隐藏；点击席位名称可单独查看';
  seat.append(meta);

  if (player.melds.length) {
    const melds = document.createElement('div');
    melds.className = 'meld-row';
    player.melds.forEach((meld) => {
      const group = document.createElement('span');
      group.className = `meld meld-${meld.kind}`;
      group.title = MELD_LABELS[meld.kind];
      meld.tiles.forEach((tile) => group.append(tileElement(tile)));
      melds.append(group);
    });
    seat.append(melds);
  }

  if (!visibleHand(player.seat)) {
    const backs = document.createElement('div');
    backs.className = 'back-row';
    for (let index = 0; index < player.hand.length; index += 1) {
      const back = document.createElement('span');
      back.className = 'tile tile-back';
      backs.append(back);
    }
    seat.append(backs);
    return;
  }

  const hand = document.createElement('div');
  hand.className = 'hand-row';
  const handTiles = [...player.hand];
  let drawnTile = null;
  if (player.drawnTile !== null) {
    const index = handTiles.lastIndexOf(player.drawnTile);
    if (index >= 0) [drawnTile] = handTiles.splice(index, 1);
  }
  const canDiscard = state.phase === 'discard' && state.currentPlayer === player.seat;
  handTiles.forEach((tile) => hand.append(tileElement(tile, { clickable: canDiscard })));
  if (drawnTile !== null) hand.append(tileElement(drawnTile, { clickable: canDiscard, drawn: true }));
  seat.append(hand);
}

function renderRivers() {
  state.players.forEach((player) => {
    const river = document.querySelector(`[data-river="${player.seat}"]`);
    river.replaceChildren();
    player.discards.slice(-18).forEach((tile, index, visible) => {
      const isLatest = state.latestDiscard?.seat === player.seat
        && index === visible.length - 1
        && state.latestDiscard.tile === tile;
      river.append(tileElement(tile, { lastDiscard: isLatest }));
    });
  });
}

function renderGold() {
  const indicator = $('#lab-gold-indicator');
  const gold = $('#lab-gold-tile');
  indicator.replaceChildren(tileElement(state.goldIndicator));
  gold.replaceChildren(tileElement(state.goldTile));
  $('#lab-gold-note').textContent = state.goldTile === 33
    ? '本局白板本身就是真金，没有另一张替身牌。'
    : `白板只按${tileName(state.goldTile)}使用，不是万能牌，也不算真金。`;
}

function renderLog() {
  const list = $('#lab-event-list');
  list.replaceChildren();
  if (!state.events.length) {
    const empty = document.createElement('li');
    empty.textContent = '暂无记录';
    list.append(empty);
    return;
  }
  state.events.forEach((event) => {
    const item = document.createElement('li');
    item.innerHTML = `<b>#${String(event.number).padStart(2, '0')} · ${event.title}</b>${event.detail}`;
    list.append(item);
  });
}

function selfKongActions() {
  if (state.phase !== 'discard') return [];
  const player = state.players[state.currentPlayer];
  const actions = [];
  [...new Set(player.hand)].forEach((tile) => {
    if (tile !== state.goldTile && player.hand.filter((item) => item === tile).length === 4) {
      actions.push({ kind: 'an_kan', seat: player.seat, tile, label: `暗杠 ${tileName(tile)}` });
    }
  });
  player.melds.forEach((meld, index) => {
    if (meld.kind === 'pong' && new Set(meld.tiles).size === 1) {
      const tile = meld.tiles[0];
      if (tile !== state.goldTile && player.hand.includes(tile)) {
        actions.push({ kind: 'add_kan', seat: player.seat, tile, meldIndex: index, label: `补杠 ${tileName(tile)}` });
      }
    }
  });
  return actions;
}

function renderActions() {
  const container = $('#lab-response-actions');
  container.replaceChildren();
  if (state.phase === 'over') {
    $('#lab-message').textContent = '本局已手动结束；可重新洗牌继续验证';
    return;
  }
  if (state.phase === 'response') {
    $('#lab-message').textContent = `${SEAT_NAMES[state.latestDiscard.seat]}打出${tileName(state.latestDiscard.tile)}；选择实际采用的响应`;
    state.pendingActions.forEach((action) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `${SEAT_NAMES[action.seat]} · ${action.label}`;
      button.title = action.consumed?.length ? `使用手牌：${action.consumed.map(tileName).join('、')}` : action.label;
      button.addEventListener('click', () => applyClaim(action));
      container.append(button);
    });
    const pass = document.createElement('button');
    pass.type = 'button';
    pass.className = 'primary';
    pass.textContent = '全部过 → 下家摸牌';
    pass.addEventListener('click', passAllClaims);
    container.append(pass);
    return;
  }
  const promptByReason = {
    claim: `${SEAT_NAMES[state.currentPlayer]}吃／碰后直接出牌；本次没有摸牌`,
    kong: `${SEAT_NAMES[state.currentPlayer]}杠后已补牌；请选择一张牌打出`,
    opening_draw: `${SEAT_NAMES[state.currentPlayer]}起手摸牌后出牌；点击该席任意一张明牌`,
    draw: `${SEAT_NAMES[state.currentPlayer]}摸牌后出牌；点击该席任意一张明牌`,
  };
  $('#lab-message').textContent = promptByReason[state.turnReason] || `${SEAT_NAMES[state.currentPlayer]}请选择一张牌打出`;
  selfKongActions().forEach((action) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = action.label;
    button.addEventListener('click', () => applySelfKong(action));
    container.append(button);
  });
}

function render() {
  state.players.forEach(renderSeat);
  renderRivers();
  renderGold();
  renderActions();
  renderLog();
  $('#lab-current-player').textContent = state.phase === 'over' ? '本局结束' : SEAT_NAMES[state.currentPlayer];
  $('#lab-wall-count').textContent = `${state.wall.length} 张`;
  $('#lab-turn-count').textContent = `第 ${state.turn} 巡`;
  $('#lab-phase').textContent = ({ discard: '出牌', response: '响应', over: '结算', opening: '发牌' })[state.phase];
  $('#lab-last-discard').textContent = state.latestDiscard
    ? `${SEAT_NAMES[state.latestDiscard.seat]}打出 ${tileName(state.latestDiscard.tile)}`
    : '等待首张弃牌';
  $$('.view-switcher button').forEach((button) => {
    button.classList.toggle('is-selected', button.dataset.view === String(state.view));
  });
}

function effectiveTile(tile) {
  return state.goldTile !== 33 && tile === 33 ? state.goldTile : tile;
}

function claimConsumptions(hand, required) {
  const results = [];
  const visit = (index, remaining, consumed) => {
    if (index === required.length) {
      results.push([...consumed].sort((a, b) => a - b));
      return;
    }
    const target = required[index];
    const candidates = target === state.goldTile
      ? (state.goldTile !== 33 ? [33] : [])
      : [target];
    [...new Set(candidates)].forEach((candidate) => {
      const found = remaining.indexOf(candidate);
      if (found < 0) return;
      const next = [...remaining];
      next.splice(found, 1);
      visit(index + 1, next, [...consumed, candidate]);
    });
  };
  visit(0, [...hand], []);
  const unique = new Map(results.map((tiles) => [tiles.join(','), tiles]));
  return [...unique.values()];
}

function chiRequirements(hand, discarded) {
  const effective = effectiveTile(discarded);
  if (!isSuited(effective)) return [];
  const base = Math.floor(effective / 9) * 9;
  const rank = effective % 9;
  const results = [];
  [[-2, -1], [-1, 1], [1, 2]].forEach((offsets) => {
    const positions = offsets.map((offset) => rank + offset);
    if (!positions.every((position) => position >= 0 && position < 9)) return;
    results.push(...claimConsumptions(hand, positions.map((position) => base + position)));
  });
  return results;
}

function collectResponseActions(discarder, discarded) {
  if (discarded === state.goldTile) return [];
  const actions = [];
  const effective = effectiveTile(discarded);
  state.players.forEach((player) => {
    if (player.seat === discarder) return;
    claimConsumptions(player.hand, [effective, effective, effective]).forEach((consumed) => {
      actions.push({ seat: player.seat, kind: 'ming_kan', consumed, label: `明杠 ${tileName(discarded)}` });
    });
    claimConsumptions(player.hand, [effective, effective]).forEach((consumed) => {
      actions.push({ seat: player.seat, kind: 'pong', consumed, label: `碰 ${tileName(discarded)}` });
    });
  });
  const nextSeat = (discarder + 1) % 4;
  chiRequirements(state.players[nextSeat].hand, discarded).forEach((consumed) => {
    const display = [...consumed.map(effectiveTile), effective].sort((a, b) => a - b).map(tileName).join('·');
    actions.push({ seat: nextSeat, kind: 'chi', consumed, label: `吃 ${display}` });
  });
  const priority = { ming_kan: 0, pong: 1, chi: 2 };
  return actions.sort((left, right) => priority[left.kind] - priority[right.kind] || left.seat - right.seat);
}

function removeTiles(hand, tiles) {
  tiles.forEach((tile) => {
    const index = hand.indexOf(tile);
    if (index >= 0) hand.splice(index, 1);
  });
}

function discardTile(tile) {
  if (state.phase !== 'discard') return;
  const player = state.players[state.currentPlayer];
  const index = player.hand.indexOf(tile);
  if (index < 0) return;
  player.hand.splice(index, 1);
  player.drawnTile = null;
  player.discards.push(tile);
  state.latestDiscard = { seat: player.seat, tile };
  addLog('弃牌', `${player.name}打出${tileName(tile)}`);
  state.pendingActions = collectResponseActions(player.seat, tile);
  if (tile === state.goldTile) {
    addLog('真金弃置', `${player.name}打出的真金不可被吃、碰、杠或点炮胡，直接轮到下家摸牌`);
    startNextTurn(player.seat);
    return;
  }
  if (state.pendingActions.length) {
    state.phase = 'response';
    addLog('等待响应', `桌上出现 ${state.pendingActions.length} 个可验证的吃、碰或杠选项`);
    render();
    return;
  }
  startNextTurn(player.seat);
}

function startNextTurn(previousSeat) {
  const nextSeat = (previousSeat + 1) % 4;
  state.currentPlayer = nextSeat;
  state.phase = 'discard';
  state.pendingActions = [];
  state.turnReason = 'draw';
  state.turn += 1;
  const tile = drawForPlayer(nextSeat);
  if (tile !== null && state.phase !== 'over') addLog('摸牌', `${SEAT_NAMES[nextSeat]}从牌墙摸进一张牌`);
  render();
}

function passAllClaims() {
  if (state.phase !== 'response') return;
  const discarder = state.latestDiscard.seat;
  addLog('全部过', `其余三家均不吃、不碰、不杠，轮到${SEAT_NAMES[(discarder + 1) % 4]}摸牌`);
  startNextTurn(discarder);
}

function applyClaim(action) {
  if (state.phase !== 'response') return;
  const discarder = state.latestDiscard.seat;
  const discarded = state.latestDiscard.tile;
  const source = state.players[discarder];
  const claimant = state.players[action.seat];
  source.discards.pop();
  removeTiles(claimant.hand, action.consumed);
  claimant.drawnTile = null;
  claimant.melds.push({ kind: action.kind, tiles: [...action.consumed, discarded] });
  state.currentPlayer = action.seat;
  state.phase = 'discard';
  state.pendingActions = [];
  state.turnReason = action.kind === 'ming_kan' ? 'kong' : 'claim';
  state.latestDiscard = null;
  addLog(MELD_LABELS[action.kind], `${claimant.name}${action.label}，取得出牌权`);
  if (action.kind === 'ming_kan') drawForPlayer(action.seat, { replacement: true });
  render();
}

function applySelfKong(action) {
  if (state.phase !== 'discard' || action.seat !== state.currentPlayer) return;
  const player = state.players[action.seat];
  if (action.kind === 'an_kan') {
    removeTiles(player.hand, [action.tile, action.tile, action.tile, action.tile]);
    player.melds.push({ kind: 'an_kan', tiles: [action.tile, action.tile, action.tile, action.tile] });
  } else {
    removeTiles(player.hand, [action.tile]);
    player.melds[action.meldIndex] = { kind: 'add_kan', tiles: [action.tile, action.tile, action.tile, action.tile] };
  }
  player.drawnTile = null;
  addLog(MELD_LABELS[action.kind], `${player.name}${action.label}，从牌墙补一张牌`);
  drawForPlayer(action.seat, { replacement: true });
  state.turnReason = 'kong';
  render();
}

function finishDraw() {
  if (!state || state.phase === 'over') return;
  state.phase = 'over';
  state.pendingActions = [];
  addLog('流局', '牌墙已摸完，本局无人胡牌');
  const result = $('#lab-result');
  result.classList.remove('hidden');
  result.innerHTML = '<div><h2>本局流局</h2><p>牌墙已摸完；点击重新洗牌可继续检查。</p></div><button class="primary-button" type="button">重新洗牌</button>';
  result.querySelector('button').addEventListener('click', newFourPlayerGame);
}

function markManualWin(winType) {
  if (state.phase === 'over') return;
  const winner = state.players[state.currentPlayer];
  state.phase = 'over';
  state.pendingActions = [];
  addLog('手动结算', `${winner.name}被标记为${winType}，主分 ${WIN_POINTS[winType]}`);
  const result = $('#lab-result');
  result.classList.remove('hidden');
  result.innerHTML = `<div><h2>${winner.name} · ${winType}</h2><p>手动验牌结果：主分 ${WIN_POINTS[winType]}。本页暂不自动核对牌形，也不计算花、杠、金等水钱。</p></div><button class="primary-button" type="button">再开一局</button>`;
  result.querySelector('button').addEventListener('click', newFourPlayerGame);
  render();
}

function setView(view) {
  state.view = view;
  render();
}

$$('.view-switcher button').forEach((button) => {
  button.addEventListener('click', () => setView(button.dataset.view));
});
$$('[data-win]').forEach((button) => {
  button.addEventListener('click', () => markManualWin(button.dataset.win));
});
$('#new-four-game').addEventListener('click', newFourPlayerGame);
$('#clear-lab-log').addEventListener('click', () => {
  state.events = [];
  renderLog();
});

newFourPlayerGame();
