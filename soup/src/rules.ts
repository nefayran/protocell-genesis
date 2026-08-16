import raw from '../../data/soup.json'

/** Один из четырёх элементарных строительных блоков — не готовый амфифил. */
export interface Monomer {
  id: string
  kind: 'carbon' | 'head' | 'donor' | 'catalyst'
  radiusSigma: number
  polar: boolean
}

/**
 * Правило образования или разрыва связи между двумя видами мономеров.
 * У каждого 'bond' обязана быть парная 'break' с ТОЙ ЖЕ энергией — это и есть
 * детальный баланс: абсолютная скорость (rank D, оценка) не откалибрована,
 * но отношение вперёд/назад калибровано энергией и поэтому честное.
 */
export interface Rule {
  id: string
  kind: 'bond' | 'break'
  a: string
  b: string
  requiresCatalyst: boolean
  energyKT: number
  attemptRate: number
  rank: 'A' | 'B' | 'C' | 'D'
  basis: string
}

export interface Sweep {
  kT: number[]
  carbonDensity: number[]
  catalystCount: number[]
}

export interface Soup {
  /** Единственная явная калибровка временнóй шкалы модели (kappa_t на экране в отчётах). */
  kappaT: number
  monomers: Monomer[]
  rules: Rule[]
  start: Record<string, number>
  sweep: Sweep
}

const ALLOWED_MONOMER_KINDS = new Set<Monomer['kind']>(['carbon', 'head', 'donor', 'catalyst'])

const REQUIRED = ['kappaT', 'monomers', 'rules', 'start', 'sweep'] as const

export function loadSoup(): Soup {
  const s = raw as unknown as Soup
  for (const key of REQUIRED) {
    if ((s as unknown as Record<string, unknown>)[key] === undefined) {
      throw new Error(`data/soup.json: отсутствует поле ${key}`)
    }
  }
  return s
}

/**
 * Отношение обратной скорости к прямой при детальном балансе для правила
 * образования связи: exp(-энергия связи в единицах kT). При acceptanceProbability
 * с kT=1 (базовая калибровка kappaT) это ровно то отношение, которое получает
 * система из связки acceptanceProbability(bond)/acceptanceProbability(break).
 */
export function forwardBackwardRatio(r: Rule): number {
  return Math.exp(-r.energyKT)
}

/** Вероятность ПОПЫТКИ реакции за шаг интегратора; dt приходит извне (engine/src/params), не хардкодится здесь. */
export function attemptProbability(r: Rule, dt: number): number {
  return r.attemptRate * dt
}

/**
 * Вероятность ПРИНЯТИЯ попытки по Метрополису. Образование связи снижает
 * энергию на energyKT (ΔE = -energyKT) и поэтому принимается всегда;
 * разрыв — та же связь в обратную сторону (ΔE = +energyKT) и принимается
 * с вероятностью exp(-energyKT/kT). Оба случая — одна формула, поэтому
 * прямое и обратное направления автоматически удовлетворяют детальному
 * балансу и не могут разъехаться по отдельным константам.
 */
export function acceptanceProbability(r: Rule, kT: number): number {
  const deltaE = r.kind === 'bond' ? -r.energyKT : r.energyKT
  return Math.min(1, Math.exp(-deltaE / kT))
}

/**
 * Проверяет три инварианта набора правил и один инвариант стартового состава.
 * Бросает Error с сообщением, называющим нарушителя, при первом нарушении:
 *  - у правила rank не 'D' (для этой модели измеренных констант скорости нет —
 *    заявлять более высокий ранг было бы нечестно);
 *  - у правила пустое или тривиальное basis (обоснование скорости обязано быть содержательным);
 *  - у правила образования ('bond') нет парного разрыва ('break') с теми же a/b,
 *    либо энергии пары не совпадают — без этого детальный баланс не определён;
 *  - у мономера вид не из четырёх элементарных ('carbon'|'head'|'donor'|'catalyst'),
 *    либо start ссылается на id, не объявленный в monomers — это и есть проверка
 *    «только мономеры, без готового амфифила» на входе.
 */
export function assertRulesConsistent(s: Soup): void {
  for (const r of s.rules) {
    if (r.rank !== 'D') {
      throw new Error(
        `data/soup.json: правило ${r.id} имеет ранг ${r.rank}, а для скоростей синтеза без измеренных констант допускается только ранг D`,
      )
    }
    if (!r.basis || r.basis.trim().length <= 10) {
      throw new Error(`data/soup.json: правило ${r.id} не имеет содержательного обоснования (basis)`)
    }
  }

  for (const r of s.rules.filter((x) => x.kind === 'bond')) {
    const back = s.rules.find((x) => x.kind === 'break' && x.a === r.a && x.b === r.b)
    if (!back) {
      throw new Error(
        `data/soup.json: у правила образования ${r.id} (${r.a}-${r.b}) нет парного правила разрыва — детальный баланс не определён`,
      )
    }
    if (back.energyKT !== r.energyKT) {
      throw new Error(
        `data/soup.json: правила ${r.id} и ${back.id} расходятся по энергии (${r.energyKT} против ${back.energyKT}) — парный разрыв обязан иметь ту же энергию, иначе детальный баланс нарушен`,
      )
    }
  }

  for (const m of s.monomers) {
    if (!ALLOWED_MONOMER_KINDS.has(m.kind)) {
      throw new Error(
        `data/soup.json: мономер ${m.id} имеет вид "${m.kind}", не входящий в набор элементарных строительных блоков`,
      )
    }
  }

  const ids = new Set(s.monomers.map((m) => m.id))
  for (const k of Object.keys(s.start)) {
    if (!ids.has(k)) {
      throw new Error(
        `data/soup.json: стартовый состав ссылается на "${k}", который не объявлен как мономер — это может быть готовый амфифил, а не строительный блок`,
      )
    }
  }
}
