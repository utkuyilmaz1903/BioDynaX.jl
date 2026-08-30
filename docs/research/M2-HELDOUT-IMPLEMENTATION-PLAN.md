---
name: M2 Held-out Implementation Plan
overview: "Onaylı V1 M2 semantiğinin M2-A…M2-H uygulama dilimleri. Kaynak koda dokunulmaz. Yeni semantik, yeni kapı, M3/M4 kancası veya public API yok."
todos:
  - id: m2-heldout-impl
    content: "M2-A…H: 7/2 split; tek onaylı eğitmen fit(split.train); dış bant train-türevli; Case B Q7 açık; ev.d_rmse_* üretim yolu; 0.30 holdout kapısı değil; geçici set mutasyonu yok"
    status: pending
isProject: false
---

# M2 Held-out Uygulama Planı

Kaynak, test, benchmark veya örnek bu belgede uygulanmaz. Bu belge
[docs/research/V1-IMPLEMENTATION-PLAN.md](docs/research/V1-IMPLEMENTATION-PLAN.md)
Milestone 2’nin **uygulama dilimidir**. Çelişide V1 kazanır. Bu
belgede V1 ile çelişen bayat semantik yoktur.

M2-A…M2-H **iç uygulama dilimleridir**. `LOCKED_PUBLIC_EXPORTS` /
semver yüzeyi değildir. Ayrı public API dilimi açılmaz.

## Belge otoritesi

Onaylı bilimsel tasarım yeniden çizilmez.

```
ExperimentSet
    ↓
ExperimentSplit
    ├── train   = (1, 2, 3, 4, 5, 6, 7)
    └── holdout = (8, 9)
```

`ExperimentSet` değişmez. Train/holdout set’in alanı değildir.
`split_experiments` onaylı API **değildir** (yasak genel splitter
adı). Onaylı internal üretici yalnız `unique_claim_experiment_split`’tir.
6/3 **yoktur** (yasak yanlış split).

Dokuz deney tek `generate_recovery_experiments` ile, split’ten
**önce** üretilir. Onaylı yol:

```
generate_recovery_experiments(...)
    ↓
unique_claim_experiment_split(set)
    ↓
fit_unknown_destruction(..., split.train)
    ↓
evaluation
```

`_train_unknown_edge` generate sayısı **tam 1**. Dönüş
`return fit, set`. İkinci generate yoktur (Y-RNG-1…Y-RNG-7).
L-RNG dar unique-claim production-path sözleşmesidir; splitter
birim testi **değildir**. Depo-geneli `generate_*` yasağı
**yoktur**. Depo-geneli RNG yasağı **yoktur**.

Split, üretilmiş `Experiment` nesnelerinin partition /
görünümüdür. Orijinal `ExperimentSet` mutasyonu yoktur (geçici
`pop!` / `insert!` / `resize!` / restore ve yardımcı dolayımı
dahil). L-SET-INTACT giriş-noktası-yalnız gövde taraması
**değildir**; geçişli saflık sözleşmesidir. Depo-geneli mutator
yasağı **yoktur**.

`fit_unknown_destruction` yalnız `split.train` alır.
Train keşif domain’i yalnız `split.train` alır.
Holdout yalnız değerlendirmedir.

Legacy `data_residual` = mevcut IC[1] residual.
Yeni M2 kanıtı ayrıdır.
Holdout 0.30 kapısı yoktur.
M3 fonksiyonel identifiability yoktur.
M4 yörünge-occupancy keşfi yoktur.
Yeni public API yoktur.
M1 composer / kontrol-akışı / rapor sözleşmeleri durur.

M1 composer keşfi (train-grid + dummy time) M2 holdout
değerlendirmesinden ve M4 occupancy keşfinden **ayrıdır**.
`_evaluate_unknown_rate_recovery` M2 holdout / split verisi
**almaz** (`holdout=` / `split=` yok). L-DISC-B, L-SET-INTACT
gibi geçişlidir (B-1 imza / B-2 yardımcı grafı / B-3 canlı
girdi). Depo-geneli keşif yasağı **yoktur**.

## Tek üretim çağrı yeri

İkinci yorum yoktur. Sahiplik için “veya”, “ya da”, “alternatif”
dili yoktur.

**Tek onaylı üretim yolu:**

```
run_recovery_suite
    → _train_unknown_edge
    → _evaluate_unknown_rate_recovery
    → mevcut identifiability
    → evaluate_holdout
    → report_recovery
```

Göreli sıra her unique-claim section’da zorunludur:

```
mevcut ident çağrısı
    → evaluate_holdout(...)
    → report_recovery(...)
```

`evaluate_holdout` şuralarda **değildir**:

- `_evaluate_unknown_rate_recovery` gövdesi
- `report_recovery` gövdesi
- residual kapanışı
- başka bir M2 yardımcısı
- `_ensure_holdout` sarmalayıcısı
- çoğaltılmış ikinci çağrı

Üretim kaynak sözleşmesi:

```
count("evaluate_holdout(", recovery_suite_section_body(:ude_discovery)) == 1
count("evaluate_holdout(", recovery_suite_section_body(:mm_unknown)) == 1
count("evaluate_holdout(", read("src/Recovery.jl")) == 2
count("evaluate_holdout(", read("src/RecoveryPipeline.jl")) == 1
```

Recovery.jl’deki 2 occurrence tanım değildir; biri `:ude_discovery`,
biri `:mm_unknown`. RecoveryPipeline.jl’deki 1 occurrence
`function evaluate_holdout` tanımıdır.

`report_recovery` **önceden hesaplanmış** holdout sonucunu alır.
`report_recovery` holdout **hesaplamaz**.
`report_recovery(..., holdout = nothing)` ⇒ `result.holdout === nothing`.
`report_recovery` Q7 kanıtı imal etmez.

`report_recovery` içinde gizli `evaluate_holdout` L-SITE kırmızısıdır.

## L-RNG — tek generate (production yolu)

`evaluate_holdout` sahipliği L-SITE’tir. Generate sahipliği
L-RNG’dir. İkisi karışmaz. L-SPLIT-ID L-RNG **değildir**.

Onaylı yol yukarıdaki generate → split → `fit(split.train)` →
evaluation dizisidir. Unique-claim production path’te ikinci
deney / veri üretimi **yoktur**.

**L-RNG kaynak sözleşmesi** (depo-geneli `generate_*` yasağı
**yoktur**):

1. `_train_unknown_edge` içinde `generate_recovery_experiments(`
   **tam bir** kez.
2. `_train_unknown_edge` dönüşü: `return fit, set`.
3. `_train_unknown_edge` içinde yok: ikinci
   `generate_recovery_experiments(`, `generate_experiment_set(`,
   `generate_data(`.
4. `_train_unknown_edge` sonrası unique-claim production path
   ek deney / veri üretmez.
5. `unique_claim_experiment_split` üretmez.
6. `evaluate_holdout` üretmez.
7. `:ude_discovery` ve `:mm_unknown` section gövdeleri üretmez.

`generate_recovery_experiments` tanımı içindeki
`generate_experiment_set(` M1 iç yoludur; ikinci üretim
**değildir**. `unique_claim_experiment_set`, örnek, honesty,
DataGen, `:partial_obs` L-RNG hedefi **değildir**.

Yasak gövde (L-RNG kırmızı; Y-RNG-1):

```julia
function _train_unknown_edge(...)
    set = generate_recovery_experiments(...)
    split = unique_claim_experiment_split(set)
    fit = fit_unknown_destruction(..., split.train)
    return fit, generate_recovery_experiments(...)
end
```

Eşdeğerleri de kırmızı: `generate_experiment_set(...)`,
`generate_data(...)`, başka data-generation yardımcısı, yerel
`_regen` / `_fresh_set` / `_second_draw` dolayımı, section /
splitter / `evaluate_holdout` içi ikinci generate.

İkinci generate paylaşılan suite RNG durumunu ilerletir.
`generate_data` `σ=0` iken bile `randn` tüketir. Bu hem ikinci
set’in gürültü çekimini hem de sonraki dummy
`consume_shared_suite_rng!` / diğer section davranışını kaydırır.
Per-section `MersenneTwister` yoktur. Yeni RNG soyutlaması yoktur.
Mevcut dummy consume durur.

## Kesin alan yüzeyleri

### ExperimentSplit

Yer: [src/RecoveryPipeline.jl](src/RecoveryPipeline.jl). Unexported.

```julia
struct ExperimentSplit
    train_indices::NTuple{7,Int}
    holdout_indices::NTuple{2,Int}
    train::ExperimentSet
    holdout::ExperimentSet
end
```

`fieldnames(ExperimentSplit)` **tam olarak**:

`(:train_indices, :holdout_indices, :train, :holdout)`

Sabitler splitter yanında; `UNIQUE_CLAIM_PROTOCOL` içinde değil:

- `UNIQUE_CLAIM_TRAIN_INDICES = (1, 2, 3, 4, 5, 6, 7)`
- `UNIQUE_CLAIM_HOLDOUT_INDICES = (8, 9)`

Tek üretici: `unique_claim_experiment_split(set::ExperimentSet) → ExperimentSplit`.
`length(set) == 9` zorunlu; aksi `ArgumentError`.

```
split.train_indices === (1, 2, 3, 4, 5, 6, 7)
split.holdout_indices === (8, 9)
split.train[i] === set.experiments[i]                      # i = 1:7
split.holdout[1] === set.experiments[8]
split.holdout[2] === set.experiments[9]
1 ∈ split.train_indices
```

Orijinal `set.experiments` üzerinde `splice!` / `deleteat!` /
`pop!` / `push!` / `insert!` / `append!` / `resize!` / `setindex!` /
`replace!` **yoktur**. Anlık görüntü eşitliği geçici mutasyonu
öldürmez. L-SET-INTACT geçişlidir: giriş noktası + erişilebilir
yerel yardımcılar. Giriş-noktası-yalnız gövde taraması
**yetersizdir**. Depo-geneli mutator yasağı **yoktur**.

### HoldoutEvidence

Yer: [src/RecoveryPipeline.jl](src/RecoveryPipeline.jl). Unexported.
Çağrıldığında `evaluate_holdout` **her zaman** bu tipi döner.

```julia
struct HoldoutEvidence
    data_residual_train::Float64
    data_residual_holdout::Float64
    d_rmse_holdout::Float64
    d_rmse_holdout_domain::Float64
end
```

`fieldnames(HoldoutEvidence)` **tam olarak**:

`(:data_residual_train, :data_residual_holdout, :d_rmse_holdout, :d_rmse_holdout_domain)`

`length == 4`. Ek alan L-FIELDS kırmızısıdır.

Yasak alanlar: `functional_identifiability`, `restart_agreement`,
`uncertainty`, `hypothesis`, occupancy çerçevesi, Q4 alanları,
Q7 catch-all, per-IC vektörleri.

### MechanismRecoveryResult

Yalnız şu iki alan eklenir:

```
split::Union{Nothing,ExperimentSplit} = nothing
holdout::Union{Nothing,HoldoutEvidence} = nothing
```

`haskey(result, :holdout)` alan varlığıdır.
Q7 kanıtı: `result.holdout !== nothing`.
`Inf` eksik `HoldoutEvidence`’ın ikinci temsili değildir.

## Kesin formüller

### Residual

`D_hat_fn` eğitilmiş nöral yıkımdır. Sembolik
`equation_to_function` **değildir** (yasak holdout \(D\) yolu).

```
ρ_i = hybrid_data_residual(
    model, params, term, D_hat_fn,
    exp.u0, (first(exp.times), last(exp.times)),
    exp.times, exp.observations;
    mask = exp.mask)
```

```
data_residual            = mevcut IC[1] legacy residual
data_residual_train      = (ρ_1 + ρ_2 + ρ_3 + ρ_4 + ρ_5 + ρ_6 + ρ_7) / 7
data_residual_holdout    = (ρ_8 + ρ_9) / 2
```

Herhangi bir karşılık gelen `ρ_i === Inf` ise ilgili agrega `Inf`.
RMS yoktur. Concat yoktur. Tek IC yoktur. Medyan yoktur.
Ağırlıklı ortalama yoktur. Aritmetik ortalama.

L-RES-HOLD zorunlu falsifikasyon (test **önce**
`ev = evaluate_holdout(...)` çağırır):

```
ρ_8 != ρ_9
ev.data_residual_holdout === (ρ_8 + ρ_9) / 2
ev.data_residual_train === (ρ_1 + ρ_2 + ρ_3 + ρ_4 + ρ_5 + ρ_6 + ρ_7) / 7
data_residual != data_residual_train
```

- Yalnız deney 8 gözlemini değiştir ⇒ `ev.data_residual_holdout`
  değişir; legacy `data_residual` değişmez; `ev.data_residual_train`
  değişmez
- Yalnız deney 1 gözlemini değiştir ⇒ legacy `data_residual`
  değişir; `ev.data_residual_holdout` değişmez
- Legacy `data_residual` mevcut IC[1] hesabıdır; nöral train
  ortalaması **değildir**

Karşılık gelen `ρ` `Inf` ise `Inf` yayılım kuralı durur.

### Holdout D

`d_rmse_holdout` =
holdout yörüngelerinde **fiilen gözlenen** regülatör koordinatlarında
nöral `D_hat` ile aynı koordinatlardaki `D_true` arasındaki göreli
RMSE.

```
r_holdout = _holdout_observed_regulators(split.holdout, term)
(R, D_hat_vals, _) = sample_unknown_destruction_grid(
    model, params, term; r_range = r_holdout, fill_value = 0.3)
d_rmse_holdout = _finite_rate_rel_rmse(D_hat_vals, truth_rate(vec(R)))
```

Sıra: deney 8’in tüm zaman sütunları, sonra deney 9; sütun sırası
korunur. Düzleştirme: `vcat` / `vec` tek 1-D `Float64`.
Normalizasyon / agregasyon / sonluluk: `_finite_rate_rel_rmse` =
mevcut `rate_rel_rmse` kuralı:

```
scale = max(sqrt(mean(abs2, truth_vec)), eps(Float64))
rate_rel_rmse = sqrt(mean(abs2, estimate − truth)) / scale
```

`estimate` veya `truth` içinde sonlu olmayan değer ⇒ `Inf`.
Holdout arayüzü `rate_rel_rmse` **değildir** (yasak eski arayüz adı);
test `ev.d_rmse_*` okur.

`d_rmse_holdout_domain` =
aynı nöral `D_hat`’in **tam** train-türevli dış bantta `D_true` ile
göreli RMSE’si.

```
r_band_external = _unique_claim_external_regulator_band(split.train, term)
(R, D_hat_vals, _) = sample_unknown_destruction_grid(
    model, params, term; r_range = r_band_external, fill_value = 0.3)
d_rmse_holdout_domain = _finite_rate_rel_rmse(D_hat_vals, truth_rate(vec(R)))
```

Sıra: `range` artan sırası. Düzleştirme / normalizasyon / sonluluk
yukarıdaki ile aynı. Holdout gözlemi bu koordinatlara girmez.

Holdout \(D\) testleri **zorunlu üretim-dönüş eşitliğidir**.
Test **önce** şunu çalıştırır:

```
ev = evaluate_holdout(...)
```

Sonra **M2 protokolünün tam üretim koordinatlarıyla** assert eder:

```
ev.d_rmse_holdout ===
    _finite_rate_rel_rmse(D_hat_holdout, D_true_holdout)
ev.d_rmse_holdout_domain ===
    _finite_rate_rel_rmse(D_hat_domain, D_true_domain)
```

`D_hat_holdout` / `D_true_holdout` =
`sample_unknown_destruction_grid(..., r_range = r_holdout, fill_value = 0.3)`
çıktısı ve `truth_rate(vec(R))`.
`D_hat_domain` / `D_true_domain` =
`sample_unknown_destruction_grid(..., r_range = r_band_external, fill_value = 0.3)`
çıktısı ve `truth_rate(vec(R))`.
`r_holdout` = holdout’ta **fiilen gözlenen** regülatör koordinatları.
`r_band_external` = kilitli **train-türevli** dış bant.

`evaluate_holdout` **kendisi** şunları kullanır:

- gerçek nöral `D_hat`
- gerçek holdout gözlenen regülatör koordinatları
- gerçek kilitli train-türevli dış-bant koordinatları

`evaluate_holdout` **kullanmaz**:

- `normalize_destruction_samples`
- `equation_to_function`
- sembolik rekonstrüksiyon
- `sample_learned_function`
- train-only ızgara
- keyfi sabit ızgara
- tam-set ızgara

Beklenen RMSE’yi bağımsız hesaplayıp gerçek `ev.d_rmse_*`
dönüşlerini assert etmeyen bir test **yetersizdir**.
Yalnız `ev.d_rmse_* > 0.5` kontrolü **yetersizdir**.
Onaylı deterministik ezber fikstürünün sayısal
falsifikasyon marjı da zorunludur (L-OVERFIT).

Üretim koordinatları `evaluate_holdout` sırasında
`sample_unknown_destruction_grid`’e fiilen geçen `r_range`’dir.

`evaluate_holdout` gövdesinde **ve** ondan erişilebilir yerel
yardımcı çağrı grafında yok:

- `normalize_destruction_samples(`  (yasak holdout \(D\) yolu)
- `equation_to_function(`           (yasak holdout \(D\) yolu)
- `sample_learned_function(`
- sembolik `D` rekonstrüksiyonu
- `discover_equations(`
- `discover_unknown_rate(`
- `discover_unknown(`
- `discover_unknown_destruction(`
- depoda halihazırda bulunan herhangi bir diğer sembolik keşif
  yardımcısı
- `_peek_holdout`
- keyfi NN ızgarası / onaylı protokolle ilgisiz sabit ızgara
- holdout \(D\) için train-only ızgara
- `evaled.success` / `discovery.success` kapısı

Bu, depo-geneli keşif yasağı **değildir**.

### Dış bant — tek formül

Yalnız `split.train` nicelikleri. Deterministik. Holdout incelenmez.
Post-hoc sıkılaştırma yoktur. Gizlenmiş sabit aralık **değildir**.

```
r_train = vcat(exp.observations[term.regulator, :]
               for exp in split.train.experiments)
r_lo_train, r_hi_train = extrema(r_train)
span_train = max(r_hi_train - r_lo_train, 0.1)
r_lo_external = r_hi_train + 0.15 * span_train
r_hi_external = r_hi_train + 0.35 * span_train
n_external    = 80
r_band_external = range(r_lo_external, r_hi_external; length = n_external)
```

Alt sınır: `r_hi_train + 0.15 * span_train`.
Üst sınır: `r_hi_train + 0.35 * span_train`.
Nokta sayısı: `80`.
Sıra: `range` artan sırası.
Koordinatlar: `collect(r_band_external)` — 80 `Float64`.
`split.holdout` bu fonksiyona argüman olarak girmez.

Yasak: `range(a, b; length = 80)` train’den bağımsız
(`range(1.65, 1.85; length = 80)` gizlenmiş sabit dahil).

Dış bant **gerçekten TRAIN verisine** bağlıdır. Her iki yön
zorunludur. Testler formülü bağımsız yeniden hesaplamaz;
üretim `evaluate_holdout`’un tükettiği koordinatları
(`sample_unknown_destruction_grid` domain `r_range`) inceler.

**A — train extrema değişince tüketilen koordinatlar değişir:**

```
ev_A = evaluate_holdout(...)          # train extrema (0.50, 1.50)
r_consumed_A = yakalanan domain r_range
collect(r_consumed_A) == collect(range(1.65, 1.85; length = 80))
ev_A.d_rmse_holdout_domain

ev_B = evaluate_holdout(...)          # train extrema (1.0, 3.0); holdout aynı
r_consumed_B = yakalanan domain r_range
collect(r_consumed_B) == collect(range(3.3, 3.7; length = 80))
r_consumed_B != r_consumed_A
```

**B — holdout extrema değişince tüketilen koordinatlar değişmez:**

```
ev_before = evaluate_holdout(...)
r_consumed_before = yakalanan domain r_range
# yalnız holdout extrema ← ±SENTINEL
ev_after = evaluate_holdout(...)
r_consumed_after = yakalanan domain r_range
collect(r_consumed_after) == collect(r_consumed_before)
ev_after.d_rmse_holdout_domain === ev_before.d_rmse_holdout_domain
```

Sabit aralık yoktur. Yalnız holdout-değişmezliği yetersizdir.

## Erken durum A / B / C

Q7 yokluğunun tek temsili: `holdout === nothing`.
`Inf`, eksik `HoldoutEvidence`’ın ikinci temsili **değildir**.

Tam olarak üç durum; A ile B **çökertilmez**:

```
A: training_ok == false
B: training_ok == true && discovery.success == false
C: training_ok == true && discovery.success == true
```

`nothing` **yalnız A** içindir. Neden: `training_ok == false`
(`evaled.discovery === nothing`, M1 kodlaması) iken
`evaluate_holdout` çağrılmaz; Q7 kanıtı yoktur. B ve C
`nothing` **atamaz**.

| | A | B | C |
|---|---|---|---|
| `training_ok` | `false` | `true` | `true` |
| `discovery` | `=== nothing` | `!== nothing`, `success == false` | `!== nothing`, `success == true` |
| composer | M1 erken NamedTuple; keşif yok; `data_residual === Inf` | dummy-time çift keşif + `evaluate_recovery` | aynı + başarılı aday |
| `evaluate_holdout` | **çağrılmaz** | **çağrılır** | **çağrılır** |
| `holdout` | `=== nothing` | `HoldoutEvidence` (`!== nothing`) | `HoldoutEvidence` (`!== nothing`) |
| Q7 alanları | yok | HoldoutEvidence sözleşmesine göre sonlu / tanımlı | aynı |
| Sembolik keşif | gerekmez / yok | **gerekmez** | vardır; Q7 girdisi değildir |
| legacy `data_residual` | `Inf` | `evaluate_recovery` (M1; fail ise `Inf` olabilir) | IC[1] hybrid |

Case B kilitli politikadır ve **Q7-görünürdür**: training_ok
doğru, keşif başarısız, `evaluate_holdout` yine çağrılır,
`holdout !== nothing`, dört `HoldoutEvidence` skaler alanı
sözleşmeye göre sonlu / tanımlıdır. Q5 keşif başarısı örtük
Q7 kapısı **değildir**.

Karar, onaylı durum geçişi dışında şunlara **bağımlı değildir**:

- `!evaled.success`
- `evaled.discovery.success`
- `!evaled.discovery.success`

Aşağıdaki gövde yasaktır:

```
if !discovery.success
    holdout = nothing
end

if !evaled.discovery.success
    holdout = nothing
end

if evaled.success == false
    holdout = nothing
end

if !evaled.success
    holdout = nothing
end

if !evaled.discovery.success
    return HoldoutEvidence(Inf, Inf, Inf, Inf)
end

HoldoutEvidence(... Inf ...)
```

`evaled.success` / `evaled.discovery.success` örtük Q7 kapısı
**değildir**. L-EARLY tam sözleşme M2-G / V1 ile aynı güçtedir.

Tek onaylı karar kuralı:

```
if evaled.discovery === nothing
    holdout = nothing
else
    holdout = evaluate_holdout(
        split, evaled, ude_model, ude_fit.params, term, truth_rate)
end
```

`success` yeni M2 kapısı değildir.

## Q5 / Q7 ayrımı

`evaluate_holdout` sembolik keşif başarısına **bağımlı değildir**.

| Kanıt | Soru | Kaynak |
|---|---|---|
| `data_residual` (legacy IC[1]) | Q1 kapısı | M1 |
| `data_residual_holdout`, `d_rmse_holdout`, `d_rmse_holdout_domain` | Q1 / Q2 / Q7 | nöral `D_hat` |
| `support_recall`, sembolik keşif | Q5 | composer |

Yasak: keşif fail ⇒ Q7 baskılama; keşif sonucu ⇒ holdout metrik
girdisi; sembolik rekonstrüksiyon ⇒ holdout \(D\).

Aynı `model` / `params` / `split` / `truth_rate` ile Case B ve
Case C `evaluate_holdout` aynı dört Q7 skalerini döner.

## Tek onaylı eğitmen yolu

M2 unique-claim üretim yolunda **tam bir** onaylı eğitmen:

```
fit_unknown_destruction(..., split.train)
```

`fit_unknown_destruction` içindeki
`train_experiments_with_warmup(..., split.train, ...)` bu tek
işlemin parçasıdır.

`_train_unknown_edge` kendisi gizli ikinci fit **yapmaz**.

`_train_unknown_edge` döndükten sonra unique-claim section
**yapmaz**:

- `train_ude(`
- `train_experiments(`
- `train_experiments_with_warmup(`
- `fit_unknown_destruction(`
- `_polish_full(`
- tam 9 deney üzerinde eşdeğer ikinci eğitim geçişi

Yasak Y2-SUITE (L-FIT-A kırmızı):

```
ude_fit, ude_set = _train_unknown_edge(...)
ude_fit = train_experiments_with_warmup(
    ude_fit.params, ude_set, ude_model; ...)
```

Gizli yardımcı eşdeğerleri de kırmızı.

Sözleşme iki katmandır:

1. unique-claim section’a dar üretim kaynak sözleşmesi
2. holdout mutasyonunun nihai eğitim sonucunu değiştiremediğini
   kanıtlayan davranışsal sentinel

Bu, depo-geneli `train_experiments*` / `train_ude` yokluğu
**değildir**. `:partial_obs`, `Training.jl`, `TrainingReuse.jl`,
`ExperimentCheckpoint.jl` meşru çağrılar içerir.
`count("train_experiments_with_warmup", repo) == 0` **yanlış
sözleşmedir**.

Üretim sözleşmesi açıkça korur:

- bir onaylı `fit_unknown_destruction(..., split.train)`
- fit sonrası tam-set eğitmen yok
- `_train_unknown_edge` gizli ikinci fit yapmaz

Kapsam: `_train_unknown_edge`, onun doğrudan M2-yeni yardımcı
zinciri, iki unique-claim section.

## Yeni soyutlamalar

- `ExperimentSplit` — `ExperimentSet` alanı veya metadata gizlemesi
  set’i split sahibi yapar (yasak).
- `unique_claim_experiment_split` — genel public splitter yasak.
  `split_experiments` onaylı API **değildir**.
- `UNIQUE_CLAIM_TRAIN_INDICES` / `UNIQUE_CLAIM_HOLDOUT_INDICES` —
  protokol NamedTuple’ına split alanı eklenmez.
- `evaluate_holdout` — `evaluate_recovery` metrik-only kalır;
  composer holdout hesaplamaz.
- `HoldoutEvidence` — Q7 skalerleri düz MRR alanı olmaz.

## Dilim şablonu

Her M2-X: amaç, dosyalar, fonksiyonlar / tipler, girdiler / çıktılar,
invariantlar, testler (V1 L-*), rollback, kaynak sözleşmeleri, yasaklar.

Derleme: `evaluate_holdout → HoldoutEvidence` olduğu için ince struct
M2-D ile aynı dosyada doğabilir; alan kilidi M2-E’dedir.

---

### M2-A — `ExperimentSplit`

Dosyalar: [src/RecoveryPipeline.jl](src/RecoveryPipeline.jl);
[test/test_holdout.jl](test/test_holdout.jl);
[test/runtests.jl](test/runtests.jl);
[test/internals.jl](test/internals.jl);
[test/test_recovery_pipeline.jl](test/test_recovery_pipeline.jl)
`!isdefined(..., :ExperimentSplit)` kilitleri **aynı dilimde**
unexported + 7/2’ye retarget.

Fonksiyon: `unique_claim_experiment_split(set::ExperimentSet) → ExperimentSplit`.

Invariantlar: uzunluk 9; `length(train)==7`; `length(holdout)==2`;
kesişim boş; birleşim `1:9`; IC[1] train’de; nesne kimliği;
`observations` orijinal matrislerdir. Generate bir kez, sonra dilim.

Yasak: `Experiments.jl` / `ExperimentSet` alanları; public splitter;
`split_experiments` (yasak ad); 6/3 (yasak split); `UNIQUE_CLAIM_PROTOCOL`
alanı; orijinal `set.experiments` üzerinde `splice!` / `deleteat!` /
`pop!` / `push!` / `insert!` / `append!` / `resize!` / `setindex!` /
`replace!` (giriş noktası **veya** ondan erişilebilir yerel yardımcı);
metadata gizleme; ikinci generate.

Testler: L-SPLIT-ID, L-SPLIT-META, L-SET-META, L-SET-INTACT,
L-API, L-FIELDS. L-RNG splitter birim testi **değildir**
(M2-B / L-RNG production path).

#### L-SET-INTACT — set bütünlüğü

V1 ile **aynı** sözleşme. İki katman **birlikte** zorunludur.
Anlık görüntü kalır. Geçişli saflık sözleşmesi geçici mutasyonu
kapatır.

##### Anlık görüntü (kalır; tek başına yetmez)

M2 **öncesi** (generate sonrası, split öncesi) anlık görüntü:

```
vec_before = set.experiments
ids = [set.experiments[i] for i in 1:9]
names_before = copy(set.state_names)
units_before = copy(set.units)
meta_before = deepcopy(set.metadata)
fp_before = experiment_fingerprint(set)
```

M2 **sonrası** (`unique_claim_experiment_split` + `evaluate_holdout`):

```
set.experiments === vec_before
length(set.experiments) == 9
all(set.experiments[i] === ids[i] for i in 1:9)
set.state_names == names_before
set.units == units_before
set.metadata == meta_before
experiment_fingerprint(set) == fp_before
!hasfield(ExperimentSet, :train)
!hasfield(ExperimentSet, :holdout)
!haskey(set.metadata, :train)
!haskey(set.metadata, :holdout)
```

before/after equality is insufficient by itself because temporary
mutation followed by restoration can leave the final state unchanged.

Anlık görüntü **yetmez**. `pop!` / `insert!` / `resize!` / restore
son görüntüyü koruyabilir. Geçişli saflık sözleşmesi bu boşluğu
kapatır.

##### Geçişli saflık sözleşmesi (AST / çağrı grafı)

Depo-geneli mutator yasağı **yoktur**. Depo-geneli AST taraması
**yoktur**. Yeni genel kaynak-string honesty envanteri **yoktur**.
`push!` token’ının depo-geneli yokluğu **yanlış sözleşmedir**.
Yeni bir vektörü comprehension / indeksleme ile kurmak yasak
değildir. Onaylı M2 yardımcılarının
(`_holdout_observed_regulators`,
`_unique_claim_external_regulator_band`, `_finite_rate_rel_rmse`,
`_mean_hybrid_residual`) yeni yerel sonuç vektörüne yazması bu
sözleşmenin hedefi **değildir**.

Kapsam yalnız M2 split / evaluation çağrı grafıdır.

**Giriş noktaları** (orijinal `ExperimentSet`’i alan veya o seti
bölen / değerlendiren iki M2 fonksiyonu):

1. `unique_claim_experiment_split`
2. `evaluate_holdout`

Sözleşme (dar AST / çağrı-grafı; jenerik kaynak-string envanteri
değil):

1. M2 giriş noktasını tanır
2. Ondan erişilebilir yerel yardımcı çağrılarını statik çözer
3. Bu yardımcı gövdelerini geçişli inceler
4. Orijinal `ExperimentSet` veya onun `set.experiments`
   koleksiyonu üzerinde yasaklı bir mutator varsa kırmızı
5. Bir-düzey veya çok-düzey yardımcı dolayımını reddeder

**Geçişli kapsam.** Her giriş noktasından erişilebilir her yerel
yardımcı ki:

- orijinal `ExperimentSet` alır, **veya**
- orijinal `set.experiments` koleksiyonunu alır, **veya**
- o orijinal seti bölmek / değerlendirmekle yükümlüdür

yasaklı mutator listesine tabidir. Giriş noktası gövdesi temiz
görünüp işi bir yardımcıya devretmek kaçış **değildir**.

**Yasaklı mutatorlar** (orijinal `ExperimentSet` / orijinal
`set.experiments` üzerinde):

- `splice!`
- `deleteat!`
- `pop!`
- `push!`
- `insert!`
- `append!`
- `resize!`
- `setindex!` / `set.experiments[i] = …`
- `replace!`

Şu yardımcılar kaçış kapağı **değildir**:

- `_carve_and_restore!`
- `_split_impl`
- `_prepare!`
- `_partition!`
- `_prepare_holdout`
- `_temporary_partition`

Öldürür: splice/delete/pop/insert/resize/append/setindex/replace/restore;
yeni `Experiment` nesnesi; metadata mutasyonu; `!hasfield` tek
başına yeşil kalan şişirme; giriş-noktası-yalnız taramanın
kaçırdığı yardımcı dolayımı. `!hasfield(ExperimentSet, :holdout)`
**yetersizdir**.

```julia
# L-SET-INTACT kırmızı — giriş noktası temiz, yardımcı kirli
function unique_claim_experiment_split(set::ExperimentSet)
    return _carve_and_restore!(set)
end
function _carve_and_restore!(set)
    held = splice!(set.experiments, 8:9)
    train = ExperimentSet(set.experiments, ...)
    append!(set.experiments, held)
    ...
end

# L-SET-INTACT kırmızı — çok-düzey dolayım
function unique_claim_experiment_split(set::ExperimentSet)
    return _prepare!(set)
end
function _prepare!(set)
    return _partition!(set)
end
function _partition!(set)
    held9 = pop!(set.experiments)
    held8 = pop!(set.experiments)
    train = ExperimentSet(set.experiments, ...)
    insert!(set.experiments, 8, held8)
    insert!(set.experiments, 9, held9)
    ...
end

# L-SET-INTACT kırmızı — evaluate_holdout bir-düzey dolayım
function evaluate_holdout(split, evaled, model, params, term, truth_rate)
    return _prepare_holdout(set, split, evaled, model, params, term, truth_rate)
end
function _prepare_holdout(set, split, ...)
    held = splice!(set.experiments, 8:9)
    ...
    append!(set.experiments, held)
    ...
end

# L-SET-INTACT kırmızı — evaluate_holdout çok-düzey dolayım
function evaluate_holdout(split, evaled, model, params, term, truth_rate)
    return _temporary_partition(set, split, ...)
end
function _temporary_partition(set, split, ...)
    held9 = pop!(set.experiments)
    held8 = pop!(set.experiments)
    insert!(set.experiments, 8, held8)
    insert!(set.experiments, 9, held9)
    ...
end

# L-SET-INTACT kırmızı — giriş noktası gövdesinde restore
held = splice!(set.experiments, 8:9)
train = ExperimentSet(set.experiments, ...)
append!(set.experiments, held)

held9 = pop!(set.experiments); held8 = pop!(set.experiments)
train = ExperimentSet(set.experiments, ...)
insert!(set.experiments, 8, held8); insert!(set.experiments, 9, held9)
```

#### L-RNG — tek üretim (production yolu)

Splitter birim testi **yetmez**. L-SPLIT-ID L-RNG **değildir**.
Observation `==` / `experiment_fingerprint` değer eşitliği
**yetmez**. Depo-geneli `generate_*` yasağı **yoktur**.
Depo-geneli RNG yasağı **yoktur**. Yeni generate çerçevesi /
yeni RNG soyutlaması **yoktur**.

Kapsam (yalnız şu siteler):

1. `_train_unknown_edge` gövdesi
2. `_train_unknown_edge`’den erişilebilir yerel yardımcılar
   (dönüş set’ini üreten / tazeleyen)
3. `:ude_discovery` ve `:mm_unknown` section gövdeleri
4. `unique_claim_experiment_split` (+ erişilebilir yerel yardımcılar)
5. `evaluate_holdout` (+ erişilebilir yerel yardımcılar)

`generate_recovery_experiments` tanımı içindeki
`generate_experiment_set(` M1 iç yoludur; ikinci üretim
**değildir**.

##### Kaynak sözleşmesi

`_train_unknown_edge` gövdesi:

- `count("generate_recovery_experiments(", body) == 1`
- `generate_experiment_set(` yok
- `generate_data(` yok
- ikinci `generate_recovery_experiments(` yok
- dönüş token’ı birebir `return fit, set`
- `return fit, generate_recovery_experiments` yok
- `return fit, generate_experiment_set` yok
- `return fit, generate_data` yok

`unique_claim_experiment_split` / `evaluate_holdout` ve onlardan
erişilebilir yerel yardımcılar: `generate_recovery_experiments(` /
`generate_experiment_set(` / `generate_data(` yok.

`recovery_suite_section_body(:ude_discovery)` ve
`recovery_suite_section_body(:mm_unknown)`: aynı generate
token’ları yok.

Giriş-noktası-yalnız `count == 1` **yetmez**: `_regen` /
`_fresh_set` / `_second_draw` gövdesindeki ikinci generate
kaçış kapağı **değildir**. Dar geçişli tarama (L-SET-INTACT
tarzı; depo-geneli değil) bu yardımcıları kapsar.

##### Davranışsal provenance (zorunlu; splitter birim testi değil)

UDE eğitimi yok. Production unique-claim yolu **enstrümante**
edilir veya doğrudan çalıştırılır:

```
generate_recovery_experiments   # tek; split’ten önce
  → unique_claim_experiment_split
  → fit_unknown_destruction(..., split.train)
  → (suite) unique_claim_experiment_split(ude_set)
  → evaluate_holdout
```

`_train_unknown_edge` yolu **ve** `_train_unknown_edge` sonrası
unique-claim section yolu (split + evaluate_holdout) birlikte
zorunludur. Yalnız `unique_claim_experiment_split` birim testi
L-RNG **değildir**.

Kanca (yeni soyutlama değil; L-FIT-A trainer kancası gibi):

- `generate_recovery_experiments` giriş / çıkış
- `fit_unknown_destruction` deney girdisi
- `evaluate_holdout` `split.holdout` okuması

Birinci (ve tek) `generate_recovery_experiments` dönüşünde:

```
ids_gen = [set.experiments[i] for i in 1:9]
n_gen == 1
length(unique(objectid.(ids_gen))) == 9
```

`_train_unknown_edge` sonrası:

```
n_gen == 1
ude_fit, ude_set = _train_unknown_edge(...)
all(ude_set.experiments[i] === ids_gen[i] for i in 1:9)
all(fit_set[i] === ids_gen[i] for i in 1:7)
```

Suite split + `evaluate_holdout` sonrası:

```
n_gen == 1
split.train[i] === ids_gen[i]                      # i = 1:7
split.holdout[1] === ids_gen[8]
split.holdout[2] === ids_gen[9]
evaluate_holdout’un okuduğu holdout[i] === ids_gen[7 + i]
```

Aşağıdaki gövde **kırmızı** kalır, `set2` gözlemleri `set1`
ile değer-eşit olsa bile:

```julia
set1 = generate_recovery_experiments(...)
split = unique_claim_experiment_split(set1)
fit = fit_unknown_destruction(..., split.train)
set2 = generate_recovery_experiments(...)
return fit, set2
```

çünkü `set2.experiments[i] !== ids_gen[i]` ve `n_gen == 2`.

`experiment_fingerprint(set1) == experiment_fingerprint(set2)`
veya `set1[i].observations == set2[i].observations` tek başına
yeşil **bırakmaz**.

##### Öldürülen gövdeler

```julia
# Y-RNG-1 — ikinci generate dönüşte
function _train_unknown_edge(...)
    set = generate_recovery_experiments(...)
    split = unique_claim_experiment_split(set)
    fit = fit_unknown_destruction(..., split.train)
    return fit, generate_recovery_experiments(...)
end

# Y-RNG-2 — generate_experiment_set ikinci üretim
function _train_unknown_edge(...)
    set = generate_recovery_experiments(...)
    split = unique_claim_experiment_split(set)
    fit = fit_unknown_destruction(..., split.train)
    return fit, generate_experiment_set(...)
end

# Y-RNG-3 — generate_data ikinci üretim
function _train_unknown_edge(...)
    set = generate_recovery_experiments(...)
    split = unique_claim_experiment_split(set)
    fit = fit_unknown_destruction(..., split.train)
    return fit, generate_data(...)
end

# Y-RNG-4 — yerel yardımcıda gizli ikinci generate
function _train_unknown_edge(...)
    set = generate_recovery_experiments(...)
    split = unique_claim_experiment_split(set)
    fit = fit_unknown_destruction(..., split.train)
    return fit, _regen(...)
end
function _regen(...)
    return generate_recovery_experiments(...)
end

# Y-RNG-5 — section’da ikinci generate
ude_fit, ude_set = _train_unknown_edge(...)
ude_set = generate_recovery_experiments(...)

# Y-RNG-6 — splitter generate
function unique_claim_experiment_split(set)
    holdout = generate_recovery_experiments(...)
    ...
end

# Y-RNG-7 — evaluate_holdout generate
function evaluate_holdout(...)
    fresh = generate_recovery_experiments(...)
    ...
end
```

##### RNG semantiği (korunur; yeniden tasarlanmaz)

- dokuz IC **bir** generate
- mevcut paylaşılan suite RNG
- mevcut dummy `consume_shared_suite_rng!`
- per-section `MersenneTwister` **yok**
- ikinci generate **yok**
- yenilenmiş deneylerin ikinci `randn` çekimi **yok**

---

### M2-B — train-only fitting

Dosya: [src/Recovery.jl](src/Recovery.jl) `_train_unknown_edge`.
`fit_unknown_destruction` imzası durur. TrainingReuse iğnesi durur.

```julia
function _train_unknown_edge(...)
    _note_train_unknown_edge()
    set = generate_recovery_experiments(...)
    split = unique_claim_experiment_split(set)
    fit = fit_unknown_destruction(..., split.train)
    return fit, set   # set = tam 9; aynı generate nesnesi
end
```

`_train_unknown_edge` generate sayısı **tam 1**.
Dönüş **yalnız** `return fit, set`.
`_train_unknown_edge` kendisi gizli ikinci fit **yapmaz**.

#### L-FIT-A — dar production fit sözleşmesi (V1 ile aynı güç)

**Tek onaylı eğitmen yolu** unique-claim için:

```
_train_unknown_edge
  → fit_unknown_destruction(..., split.train)
      → train_experiments_with_warmup(..., split.train, ...)
```

Sonra: tam-set eğitim yok; `train_ude(` yok; `train_experiments(`
yok; `train_experiments_with_warmup(..., set, ...)` yok; eşdeğer
gizli eğitmen yok; `_polish_full(..., set, ...)` yok; dokuz deney
üzerinde ikinci optimizer/eğitim geçişi yok.

Bu, depo-geneli `train_experiments*` / `train_ude` yokluğu
**değildir**. `count("train_experiments_with_warmup", repo) == 0`
**yanlış sözleşmedir**.

Kapsam (dar production-path; genel envanter yok):

1. `_train_unknown_edge` gövdesi
2. `_train_unknown_edge`’in doğrudan M2-yeni yardımcı zinciri
   (`generate_recovery_experiments`, `unique_claim_experiment_split`,
   `fit_unknown_destruction`, `_note_train_unknown_edge` hariç —
   `fit_unknown_destruction` içindeki onaylı warmup durur)
3. `:ude_discovery` ve `:mm_unknown` section gövdeleri

`_train_unknown_edge` gövdesi:

- `fit_unknown_destruction(` tam **bir** kez
- o çağrının deney argümanı token’ı `split.train`
- `fit_unknown_destruction(..., set)` yok
- bu çağrıdan sonra `train_ude(` / `train_experiments_with_warmup` /
  `train_experiments(` / `_polish_full` / ikinci
  `fit_unknown_destruction` yok

`recovery_suite_section_body(:ude_discovery)` ve
`recovery_suite_section_body(:mm_unknown)`
(`_train_unknown_edge` **döndükten sonra**):

- `train_ude(` yok
- `train_experiments(` yok
- `train_experiments_with_warmup` yok
- `fit_unknown_destruction(` yok
- `_polish_full(` yok
- tam 9 deney üzerinde eşdeğer ikinci eğitim geçişi yok

Davranışsal sentinel (UDE eğitimi yok; trainer kancası):

unique-claim eğitim yolu boyunca `fit_unknown_destruction` /
`train_experiments` / `train_experiments_with_warmup` /
`train_ude` girişleri kaydedilir. Tam olarak **bir** fitting
işlemi vardır: `fit_unknown_destruction(..., split.train)`
(içindeki warmup bu tek işlemin parçasıdır; ikinci sayılmaz).
Bu kayıttan sonra `set` (9 IC) veya holdout nesnesi taşıyan
trainer girişi **0**’dır. Holdout `SENTINEL` mutasyonu hiçbir
trainer setinde görünmez ve nihai training sonucunu
değiştiremez.

`length(split.train) == 7` tek başına yetmez.

Yasak gövde kırmızı (L-FIT-A):

```
# Y1 — tam-set fit
fit = fit_unknown_destruction(..., set)

# Y2 — önce train, sonra tam set
fit = fit_unknown_destruction(..., split.train)
fit = train_experiments_with_warmup(..., set, ude_model, ...)

# Y2-POLISH — onaylı fit doğru, gizli tam-set cilası
fit = fit_unknown_destruction(..., split.train)
fit = _polish_full(..., set, ...)

# Y2-SUITE — _train_unknown_edge doğru, suite ikinci eğitmen
ude_fit, ude_set = _train_unknown_edge(...)
ude_fit = train_experiments_with_warmup(
    ude_fit.params, ude_set, ude_model; ...)
```

Yanlış gövde kırmızı (L-RNG, Y-RNG-1):

```
function _train_unknown_edge(...)
    set = generate_recovery_experiments(...)
    split = unique_claim_experiment_split(set)
    fit = fit_unknown_destruction(..., split.train)
    return fit, generate_recovery_experiments(...)
end
```

Warmup `first(split.train)` = IC[1]. Adam 7 minibatch.

Test: L-FIT-A (dar kaynak + davranışsal sentinel), L-FIT-B,
L-RNG (kaynak `count == 1` + davranışsal provenance).

---

### M2-C — train-derived regulator domain

Dosya: [src/Recovery.jl](src/Recovery.jl) `:ude_discovery` ve
`:mm_unknown`. Composer imzası durur. `ExperimentSplit` /
`HoldoutEvidence` / `split` / `holdout` almaz. `holdout=` /
`split=` anahtarı **yoktur** (L-DISC-B-1). Composer Q7 sahibi
**değil**.

```
r_range = _regulator_grid(split.train, term)
```

`data_residual_fn` hâlâ `ref_exp = first(ude_set.experiments)` (IC[1]).

Yanlış: `_regulator_grid(ude_set, term)`; holdout extrema `union`;
token sonrası overwrite.

Test: L-DOM-A, L-DOM-B, L-BAND. Sentinel fikstür zorunlu.
L-DOM-A / L-DOM-B aşağıda V1 ile **aynı güçtedir**; kısaltılmaz.

#### L-DOM-A — dar production domain sözleşmesi

Yalnızca unique-claim `:ude_discovery` / `:mm_unknown` section
gövdeleri (genel envanter yok; depo-geneli `_regulator_grid`
yasağı **yoktur**):

Gerçek production ataması **zorunludur**:

```
r_range = _regulator_grid(split.train, term)
```

- `r_range =` tam **bir** kez
- sağ taraf **birebir** `_regulator_grid(split.train, term)`
- production path sonra başka bir domain ile **değiştirmez**
- production path domain’i şunlardan **türetmez**:
  `ude_set`, `set`, `holdout`, `union(...)`, holdout extrema,
  tam-set extrema
- `r_range` post-hoc overwrite **yoktur**

`_evaluate_unknown_rate_recovery` çağrısının `r_range` anahtar
argümanı **birebir** `_regulator_grid(split.train, term)` olur
(ara değişken yoksa inline aynı token; ara değişken varsa tek
atama yukarıdaki token’dır).

Öldürür:

```
# Y4 — tam set
r_range = _regulator_grid(ude_set, term)

# Y4b — set
r_range = _regulator_grid(set, term)

# Y5 — holdout extrema birleşimi
r_train = _regulator_grid(split.train, term)
r_holdout = _regulator_grid(split.holdout, term)
r_range = union(r_train, r_holdout)

# Y6 — token doğru, sonra overwrite / set
r_range = _regulator_grid(split.train, term)
r_range = _regulator_grid(set, term)

# Y6b — token doğru, sonra union
r_range = _regulator_grid(split.train, term)
r_range = union(r_range, holdout_range)

# Y6c — token doğru, sonra holdout extrema
r_range = _regulator_grid(split.train, term)
r_range = range(min(...holdout...), max(...holdout...))

# Y6d — token doğru, sonra inline tam-set / holdout extrema
r_range = _regulator_grid(split.train, term)
lo, hi = extrema(vcat((e.observations[term.regulator, :]
                       for e in split.holdout)...))
r_range = range(min(first(r_range), lo), max(last(r_range), hi); length = 80)
```

Tek başına `_regulator_grid(ude_set, term)` /
`_regulator_grid(set, term)` literal’inin yokluğu Y6 / Y6b /
Y6c / Y6d’yi öldürmez. `r_range =` sayısı + birebir token +
sonra overwrite yokluğu zorunludur.

#### L-DOM-B — holdout extrema production domain’i değiştiremez

Gerçek IC 8–9 train kutusunda kalabilir; **sentinel fikstür
zorunludur**. Kapsam: M2 unique-claim production
(`:ude_discovery` / `:mm_unknown`).

**Bağımsız `_regulator_grid(split.train, term)` yeniden hesabı
tek başına yetmez.** `_regulator_grid(set, term)` literal
yokluğu tek başına yetmez.

Test **production yolunun tükettiği** keşif domain’ini gözler:
M1 composer’a fiilen forwarded `r_range` anahtar argümanı
**veya** M1 keşif yolunun (`sample_destruction` /
`discover_unknown_rate`) fiilen kullandığı koordinatlar.

1. Unique-claim production path’i çalıştır (section gövdesi
   veya o gövdenin domain + composer adımı). Yakala:
   `r_consumed_0` = composer’a forwarded `r_range` **veya**
   keşif yolunun tükettiği regulator koordinatları.
2. **Yalnız** holdout `observations[term.regulator, :]` ←
   `SENTINEL` (aşırı uç sentinel). Train gözlemleri değişmez.
3. Aynı production path’i tekrar çalıştır. Yakala `r_consumed_1`.
4. `collect(r_consumed_1) == collect(r_consumed_0)` bit-eşit.
5. Tam-set extrema farklı domain üretir; sentinel “tesadüfen
   önemsiz” diye yeşil kalamaz:

```
r_full = _regulator_grid(set, term)
collect(r_full) != collect(r_consumed_1)
```

Production `r_range` L-DOM-A token’ı olduğu için `r_consumed`
o token’ın değeridir. Test token’ı bağımsız yeniden hesaplayıp
“beklenen” diye yazmak yetmez; yakalanan production argüman /
keşif koordinatı zorunludur.

Öldürür: Y5 union; Y6 / Y6b / Y6c / Y6d overwrite; tam-set
extrema cache (`_regulator_grid` her iki argümanda aynı cached
değeri döner); holdout closure; yalnız bağımsız
`_regulator_grid(split.train, term)` karşılaştırması ile yeşil
kalan gövde; yalnız `_regulator_grid(set, term)` yokluğu ile
yeşil kalan overwrite.

#### L-BAND — dış bant train’den türer (her iki yön)

Yalnız holdout-değişmezliği **yetersizdir**. Sabit
`range(1.65, 1.85; length = 80)` o testi yeşil bırakır.

Üretim D-domain koordinatları, testin formülü yeniden yazması değil,
`evaluate_holdout` sırasında `sample_unknown_destruction_grid`’e
fiilen geçen `r_range`’dir. Test bu çağrıları yakalar. Domain
çağrısının `r_range`’i = `r_consumed`.

`evaluate_holdout` gövdesinde domain
`sample_unknown_destruction_grid` çağrısının `r_range` argümanı
`_unique_claim_external_regulator_band(split.train, term)` token’ıdır.

**L-BAND-TRAIN** — train extrema değişince üretim koordinatları
değişir (UDE eğitimi yok):

```
# kutu A: train extrema (0.50, 1.50)
ev_A = evaluate_holdout(split_A, evaled, model, params, term, truth_rate)
r_consumed_A = yakalanan domain r_range
collect(r_consumed_A) == collect(range(1.65, 1.85; length = 80))
ev_A.d_rmse_holdout_domain  # üretim dönüşü; bağımsız formül değil

# kutu B: train extrema (1.0, 3.0); holdout aynı
ev_B = evaluate_holdout(split_B, evaled, model, params, term, truth_rate)
r_consumed_B = yakalanan domain r_range
collect(r_consumed_B) == collect(range(3.3, 3.7; length = 80))
r_consumed_B != r_consumed_A
```

Kutu B kilitli çıktısı: `span = 2`,
`r_lo = 3.0 + 0.15 * 2 = 3.3`, `r_hi = 3.0 + 0.35 * 2 = 3.7`.
`range(r_hi + 0.15, r_hi + 0.35)` (span yok) kırmızı olur
(`3.15:3.35 ≠ 3.3:3.7`).

**L-BAND-HOLDOUT** — holdout extrema değişince üretim koordinatları
değişmez:

```
ev_before = evaluate_holdout(...)
r_consumed_before = yakalanan domain r_range
# yalnız holdout extrema ← ±SENTINEL
ev_after = evaluate_holdout(...)
r_consumed_after = yakalanan domain r_range
collect(r_consumed_after) == collect(r_consumed_before)
ev_after.d_rmse_holdout_domain === ev_before.d_rmse_holdout_domain
```

Test `ev.d_rmse_holdout_domain` okumadan “bant değişmedi / değişti”
iddia **edemez**. Test, bandı formülden bağımsız yeniden türetmez;
`r_consumed` üretim yolundan gelir.

Öldürür: holdout extrema bandı; holdout `union`; post-hoc domain;
tam-set extrema metadata / cache; keyfi sabit aralık
(`range(1.65, 1.85)` gizlenmiş sabit dahil); span’sız
`r_hi + 0.15`; yalnız L-BAND-HOLDOUT ile yeşil kalan sabit gövde.

---

### M2-D — `evaluate_holdout`

Dosya: [src/RecoveryPipeline.jl](src/RecoveryPipeline.jl).

```
evaluate_holdout(split, evaled, model, params, term, truth_rate) → HoldoutEvidence
```

Çağrıldığında her zaman `HoldoutEvidence`. `ident` almaz.
`evaled` / `ident` mutasyona uğratmaz. Keşif aşaması **değildir**.
`evaled.discovery` / `evaled.success` metrik girdisi değildir.

`D_hat` yolu: `sample_unknown_destruction_grid` /
`sample_unknown_destruction` / `_destruction_contribution`.

#### L-DISC-A — geçişli keşif yasağı (V1 ile aynı güç)

Yasak **yalnız** `function evaluate_holdout` gövdesine sınırlı
**değildir**. Yasak, `evaluate_holdout`’tan erişilebilir **yerel
yardımcı çağrı grafının tamamını** kapsar. Depo-geneli keşif
yasağı **yoktur**. Unique-claim composer’ın mevcut M1 dummy-time
çift `discover_unknown_rate` yolu **değişmez**.

Üç katman **birlikte** zorunludur.

**1. Tanım gövdesi + erişilebilir yerel yardımcı grafı**
(`function evaluate_holdout` … sonraki `\nfunction `, dört M2
yardımcısı, ve `evaluate_holdout`’tan statik olarak erişilebilir
her yerel yardımcı):

- `discover_unknown_rate(` yok
- `discover_unknown(` yok (bu token `discover_unknown_rate` öneki
  değildir; ayrı çağrı adı olarak yasaktır)
- `discover_equations(` yok
- `discover_unknown_destruction(` yok
- depoda halihazırda bulunan herhangi bir diğer sembolik keşif
  yardımcısı yok
- `_peek_holdout` yok
- `normalize_destruction_samples(` yok
- `equation_to_function(` yok
- `sample_learned_function(` yok
- `evaluate_recovery(` yok
- `_ensure_holdout` yok
- holdout \(D\) için train-only / keyfi sabit ızgara yok
- `success` / `discovery.success` kapısı yok
- holdout metriği `> 0.30` kapısı yok
- yeni keşif yolu yok

Giriş-noktası-yalnız gövde taraması **yetersizdir**.
`evaluate_holdout` gövdesi temiz görünüp keşfi bir yardımcıya
devretmek kaçış **değildir**.

**2. RecoveryPipeline.jl dosya sözleşmesi** (`evaluate_recovery` M1
`equation_to_function` kullanmaya devam eder; keşif çağırmaz):

```
count("discover_equations(", file) == 0
count("discover_unknown_rate(", file) == 0
count("discover_unknown_destruction(", file) == 0
count("_peek_holdout", file) == 0
count("evaluate_holdout(", file) == 1
```

**3. Canlı yürütme.** Test **mutlaka** `ev = evaluate_holdout(...)`
çağırır. Bu çağrının **gerçek değerlendirme yolunda** aşağıdaki
giriş noktalarının **hiçbirine** girilmez:

- `discover_unknown_rate(`
- `discover_unknown(`
- `discover_equations(`
- `discover_unknown_destruction(`
- depoda halihazırda bulunan herhangi bir diğer sembolik keşif
  yardımcısı

Composer ayrı çalıştırıldığında beklenen iki `discover_unknown_rate`
durur.

Öldürür:

```
evaluate_holdout(...)
    → _peek_holdout(...)
        → discover_equations(...)

evaluate_holdout(...)
    → helper(...)
        → discover_unknown_destruction(...)
```

#### L-DISC-B — holdout keşif girdisi değildir (geçişli)

Üç katman **birlikte** zorunludur. Anlık “composer holdout
okumaz” cümlesi yetmez. Geçişli AST / çağrı-grafı sözleşmesi
yardımcı dolayımını kapatır. L-SET-INTACT ile **aynı güç**:
giriş noktası temiz görünüp işi bir yardımcıya devretmek kaçış
**değildir**.

Kapsam: unique-claim composer **keşif çağrı grafı**.
Depo-geneli keşif yasağı **değildir**. Depo-geneli AST taraması
**yoktur**. Yeni genel kaynak-string honesty envanteri **yoktur**.

**Giriş noktası (ENTRY):**

1. `_evaluate_unknown_rate_recovery`

**Keşif-erişilebilir altgraf.** Composer’dan, aşağıdaki
sembolik-keşif giriş noktalarına **kadar ve o çağrılar
sırasında** geçişli erişilebilir her yerel yardımcı:

- `discover_unknown_rate`
- `discover_unknown`
- `discover_equations`
- `discover_unknown_destruction`
- eşdeğer sembolik-keşif giriş noktaları

Onaylı M1 composer **aynı** M1 composer’dır. Public / private
imza ve sahiplik M2 için **genişlemez**. Composer Q7 sahibi
**değil**.

`_evaluate_unknown_rate_recovery` **almaz**:

- `ExperimentSplit`
- `HoldoutEvidence`
- `split`
- `holdout`
- holdout deneyleri
- holdout gözlemleri
- holdout zamanları
- holdout türevleri
- herhangi bir M2 holdout nesnesi

`holdout=` anahtarı **eklenmez**. `split=` anahtarı **eklenmez**.

M2 holdout değerlendirmesi composer’dan **sonra** ve mevcut
identifiability çağrısından **sonra** olur. Tek geçerli kenar:

```
ident → evaluate_holdout → report_recovery
```

Yasak kenar:

```
composer → holdout
```

Onaylı kural (değişmez):

- Mevcut keşif dizisi değişmez: sample learned D →
  `training_ok` → erken dönüş **veya** dummy-time keşif →
  normalizasyon → ikinci keşif → `evaluate_recovery`.
- M2 holdout verisi o keşif dizisine **girmez**.
- M2 composer’ı holdout / occupancy keşif aşamasına **çevirmez**.
- M2 dummy-time fonksiyon-regresyon semantiğini **kaldırmaz**.
- M4 yörünge-occupancy keşfi M2’de **yoktur**.

Üç yol **ayrı** durur:

```
M1 composer keşfi:
    train-derived r_range
    +
    öğrenilmiş D örnekleri
    +
    dummy time
        times = collect(range(0.0, 1.0; length = length(r)))
    +
    unique_claim_discovery_config()
    +
    ikinci keşif için normalize öğrenilmiş D

M2 holdout değerlendirme:
    öğrenilmiş nöral D
    +
    gerçek holdout regülatör koordinatları
    +
    held-out residual
    evaluate_holdout; keşif aşaması değildir

M4 gelecek occupancy keşfi:
    yörünge-occupancy keşfi
    M2’DE UYGULANMAZ
```

L-DISC-B, L-DOM-A/B **değildir**. L-DOM-A/B train-derived
`r_range`’i yönetir:

```
r_range = _regulator_grid(split.train, term)
```

Composer o train-derived `r_range`’i alır. L-DISC-B composer’ın
o `r_range` ve öğrenilmiş D ile **ne yaptığını** yönetir:

```
split.train
    → r_range
    → composer
    → dummy-time keşif
```

**değil:**

```
split.holdout
    → composer
    → keşif
```

Onaylı M1 keşif girdileri **yalnız**:

- `R_grid`
- öğrenilmiş `D_nn`
- `term`
- dummy `times = collect(range(0.0, 1.0; length = length(r)))`
- `unique_claim_discovery_config()`
- ikinci keşif için normalize öğrenilmiş D

Keşif yolu **almaz**:

- `split`
- `split.holdout`
- `holdout`
- `holdout.observations`
- `holdout.times`
- holdout türevleri
- holdout deneyleri içeren herhangi bir `ExperimentSet`

`discover_unknown_rate(` sayacı `== 2` tek başına yetmez. İki
normal keşif çağrısı dururken üçüncü / sızdırılmış holdout
çağrısı kırmızı kalır. `training_ok == false` erken çıkış
L-DISC-B’yi yeşil **bırakamaz**.

Composer gövdesinde `evaluate_holdout` yoktur (L-SITE).
`evaluate_holdout` keşif çağırmaz (L-DISC-A). L-DISC-B, L-DISC-A
değildir: L-DISC-A `evaluate_holdout` → keşif sızıntısıdır;
L-DISC-B composer keşif grafının holdout almasıdır.

##### L-DISC-B-1 — imza / çağrı-yeri sözleşmesi

Kapsam: M2 unique-claim production section gövdeleri
`:ude_discovery` ve `:mm_unknown`. Depo-geneli tarama **yoktur**.

```
ude = recovery_suite_section_body(:ude_discovery)
mm  = recovery_suite_section_body(:mm_unknown)
count("_evaluate_unknown_rate_recovery(", ude) == 1
count("_evaluate_unknown_rate_recovery(", mm) == 1
```

Her çağrı-yeri için **zorunlu**:

- `_evaluate_unknown_rate_recovery` **tam bir** kez
- `holdout=` / `holdout =` anahtarı **yok**
- `split=` / `split =` anahtarı **yok**
- `ExperimentSplit` geçirilmez (konumsal veya anahtar)
- `HoldoutEvidence` geçirilmez (konumsal veya anahtar)
- Yalnız mevcut M1 composer girdileri geçer: konumsal
  `ude_model`, `ude_params`, `term`, `truth_rate`; anahtar
  `order`, `family`, `noise_σ`, `r_range`, `data_residual_fn`
- `r_range` L-DOM-A token’ıdır
- `data_residual_fn` M1 IC[1] kapanışıdır; `split.holdout` /
  `holdout` / deney 8 / deney 9 **yakalamaz**

Yalnız “holdout sızmamalı” cümlesi bu katmanı yeşil
**bırakamaz**. Çağrı-yeri / imza sözleşmesi kırmızı yapmak
zorundadır.

Öldürür (L-DISC-B-1 kırmızı; `r_range` token’ı doğru olsa bile):

```
# WRONG IMPLEMENTATION #1
evaled = _evaluate_unknown_rate_recovery(
    ...;
    r_range = _regulator_grid(split.train, term),
    holdout = split.holdout)

evaled = _evaluate_unknown_rate_recovery(
    ...;
    r_range = _regulator_grid(split.train, term),
    split = split)

# WRONG IMPLEMENTATION #5 — residual kapanışı kaçış değildir
evaled = _evaluate_unknown_rate_recovery(
    ...;
    data_residual_fn = d_hat -> something_using(split.holdout))
```

##### L-DISC-B-2 — geçişli AST / çağrı-grafı sözleşmesi

L-SET-INTACT ile **aynı güçte** geçişlilik. Bir-düzey tarama
**yetersizdir**.

Sözleşme (dar AST / çağrı-grafı; jenerik kaynak-string
envanteri değil):

1. Composer giriş noktasını tanır
   (`_evaluate_unknown_rate_recovery`)
2. Ondan keşif giriş noktalarına kadar erişilebilir yerel
   yardımcı çağrılarını statik çözer
3. Bu yardımcı gövdelerini **geçişli** inceler
4. Yasaklı holdout erişimi / holdout-türevli keşif girdisi
   varsa kırmızı
5. Bir-düzey veya çok-düzey yardımcı dolayımını reddeder

Keşif-erişilebilir her yerel yardımcı için **yasak**:

- `ExperimentSplit` kabul etmek
- `HoldoutEvidence` kabul etmek
- `split` kabul etmek
- `holdout` kabul etmek
- `.holdout` erişmek
- `split.holdout` erişmek
- holdout deney gözlemlerine erişmek
- holdout deney zamanlarına erişmek
- holdout türevlerine erişmek
- keşif için deney 8 veya 9 okumak
- keşfe iletilen holdout-türevli vektör almak
- yukarıdakilerden birini yapan başka bir yardımcıyı çağırmak
- `discover_unknown_rate` / `discover_unknown` /
  `discover_equations` / `discover_unknown_destruction` /
  eşdeğer keşif giriş noktasına holdout-türevli girdi geçirmek
- mevcut M1 dummy-time `discover_unknown_rate` çiftinin
  yerine / yanına `discover_unknown_destruction(` veya
  `discover_equations(` ile holdout keşif yolu açmak

Öldürür (L-DISC-B-2 kırmızı; `evaluate_holdout` keşif
çağırmasa bile; composer gövdesi temiz görünse bile):

```
# WRONG IMPLEMENTATION #2 — bir-düzey yardımcı
_evaluate_unknown_rate_recovery(...)
    → _composer_helper(...)
        → discover_unknown_rate(...,
                                split.holdout.observations, ...)

# WRONG IMPLEMENTATION #3 — holdout times / türev
_evaluate_unknown_rate_recovery(...)
    → _helper(...)
        → discover_equations(
               holdout.times,
               holdout.derivatives,
               ...)

# WRONG IMPLEMENTATION #4 — holdout keşif yolu
_evaluate_unknown_rate_recovery(...)
    → helper(...)
        → discover_unknown_destruction(...)

# WRONG IMPLEMENTATION — çok-düzey dolayım
_evaluate_unknown_rate_recovery(...)
    → _prepare_disc(...)
        → _forward_holdout(...)
            → discover_unknown_rate(... holdout ...)
```

##### L-DISC-B-3 — üretim veri-akışı testi

**Canlı üretim yolu.** Test unique-claim composer’ı
(`_evaluate_unknown_rate_recovery`) gerçek production
`r_range` token’ı ile çalıştırır ve o çağrının keşif
yolunda **fiilen geçen** keşif girdilerini yakalar
(keşif giriş noktalarını enstrümante eder veya production
forwarding’i gözler). `training_ok == true` dalı kullanılır
(ucuz fikstür; tam unique-claim UDE eğitimi yoktur).
Bağımsız paralel formül yeniden hesabı yetmez.

Yakalanan keşif çağrıları (M1 girdileri):

- `times` = mevcut dummy
  `collect(range(0.0, 1.0; length = length(r)))`;
  holdout `times` **değil**
- durum / gözlem / `R` argümanı train-grid `R_grid`;
  holdout `observations` **değil**
- `derivatives` train-grid `D_nn` (veya normalize eşdeğeri);
  holdout türevleri **değil**
- `config` = `unique_claim_discovery_config()`

**Sentinel.** Split **sonrası**, **yalnız** holdout
`observations` / `times` / türevler aşırı uç sentinel
değere mutasyona uğratılır. Train gözlemleri / zamanları
değişmez. Aynı unique-claim composer production path tekrar
çalışır.

Yakalanan gerçek composer keşif girdileri (veya onların
gerçek production sonucu) **bit-eşittir**. Test, keşif
domain’ini / girdilerini bağımsız formülle yeniden
hesaplayıp “beklenen” diye yazarak yeşil **kalamaz**.

L-DOM-B `r_range` tüketimini gözler. L-DISC-B-3 `r_range`
sonrası composer keşif girdilerini (`R_grid`, dummy
`times`, `D_nn` / normalize D, config) gözler. İkisi
karışmaz.

#### L-D-OCC — `d_rmse_holdout` gerçek holdout \(r\) (V1 ile aynı güç)

`evaluate_holdout` gövdesi occupancy
`sample_unknown_destruction_grid` çağrısının `r_range` argümanı
`_holdout_observed_regulators(split.holdout, term)` token’ıdır.

Test **yalnızca** `_finite_rate_rel_rmse(...)`’i bağımsız hesaplayıp
`ev.d_rmse_holdout` okumadan yeşil **kalamaz**.
Yalnız `ev.d_rmse_holdout > 0.5` / `ev.d_rmse_holdout_domain > 0.5`
**yetersizdir**. Bağımsız beklenen RMSE, gerçek `ev.d_rmse_*`
dönüşlerini assert etmeden **yetersizdir**.

Zorunlu canlı bağ:

```
ev = evaluate_holdout(split, evaled, model, params, term, truth_rate)
r_holdout = _holdout_observed_regulators(split.holdout, term)
(R, D_hat_vals, _) = sample_unknown_destruction_grid(
    model, params, term; r_range = r_holdout, fill_value = 0.3)
expected = _finite_rate_rel_rmse(D_hat_vals, truth_rate(vec(R)))
ev.d_rmse_holdout === expected
```

`expected` **normalize edilmemiş** nöral örneklemedir.
`normalize_destruction_samples` / `equation_to_function` /
`sample_learned_function` / keyfi ızgara / train-only ızgara /
tam-set ızgara **yoktur**.

Canlı test **her zaman** `ev = evaluate_holdout(...)` çağırır ve
`ev.d_rmse_holdout` / `ev.d_rmse_holdout_domain` /
`ev.data_residual_holdout` okur.

L-SET-INTACT: `evaluate_holdout` ikinci giriş noktasıdır. Ondan
erişilebilir yerel yardımcılar (`_prepare_holdout`,
`_temporary_partition` dahil) orijinal `ExperimentSet` /
`set.experiments` üzerinde yasaklı mutator kullanamaz.
Giriş-noktası-yalnız gövde taraması **yetersizdir**. Depo-geneli
mutator yasağı **yoktur**.

Fonksiyon tanımı M2-D’dedir. Tek üretim çağrı yeri M2-F’de
kablolanır. M2-D alternatif sahip açmaz. Birim test doğrudan
fonksiyonu çağırır; production sıra sözleşmesi L-SITE’tir.

Test: L-DISC-A, L-DISC-B (B-1/B-2/B-3), L-BAND, L-D-OCC,
L-RES-HOLD, L-OVERFIT, L-EARLY (Case B), L-GATE.

---

### M2-E — `HoldoutEvidence` alan kilidi

Dosya: [src/RecoveryPipeline.jl](src/RecoveryPipeline.jl).
Occupancy / uncertainty / hypothesis / Q4 çerçevesi değildir.
Düz MRR alanı değildir.

Test: L-FIELDS, L-M34.

---

### M2-F — `MechanismRecoveryResult` / `report_recovery` / tek çağrı yeri

Dosyalar: [src/RecoveryPipeline.jl](src/RecoveryPipeline.jl);
[src/Recovery.jl](src/Recovery.jl) unique-claim kabuğu.

```
report_recovery(evaled, ident; model, params, experiments,
                split = nothing, holdout = nothing)
```

`report_recovery` `evaluate_holdout` çağırmaz.
`_ensure_holdout` yoktur.
`HoldoutEvidence` üretmez.
KPI / `protocol_result` / stdout IC[1] `data_residual` okur.

Suite kararı (ident’den sonra, rapordan önce) her iki section’da
yukarıdaki tek `if evaled.discovery === nothing` bloğudur.
`evaled.success == false` atlama koşulu değildir.

M1 `!isdefined(ExperimentSplit)` / `:holdout ∉ fields` kilitleri
**silinmeden** retarget: unexported + 7/2; `haskey` vs `!== nothing`.
Canlı `ude_adam=0`: durum A.

Test: L-SITE, L-EARLY, L-GATE, L-API.

L-SITE öldürür:

```
function report_recovery(..., holdout = nothing)
    holdout = _ensure_holdout(holdout, evaled, ...)
end
```

---

### M2-G — sızıntı / overfitting

Dosya: [test/test_holdout.jl](test/test_holdout.jl) (hızlı, UDE
eğitimsiz). L-DISC-A, L-DISC-B-1/2/3, L-BAND, L-D-OCC, L-OVERFIT,
L-RES-LEGACY, L-RES-HOLD, L-EARLY, L-GATE, L-SET-INTACT, L-FIT-A,
L-RNG.

L-RNG **mutlaka** production unique-claim yolunu enstrümante
eder veya çalıştırır (`_train_unknown_edge` + suite split +
`evaluate_holdout`). Splitter birim testi L-RNG **değildir**.
`ids_gen[i] ===` eğitim ve holdout değerlendirme nesneleri;
`n_gen == 1`. `set2` gözlem-eşit ikinci generate kırmızı kalır.

#### L-OVERFIT — deterministik ezber (V1 ile aynı güç)

UDE **eğitimi yoktur**. Test **mutlaka** production
`evaluate_holdout(...)` çağırır ve `ev.d_rmse_holdout` /
`ev.d_rmse_holdout_domain` okur. Test-içi bağımsız
`_finite_rate_rel_rmse(...)` `ev.*` yerine **geçemez**.
Yalnız `ev.d_rmse_* > 0.5` **yetersizdir**. Bağımsız beklenen
RMSE, gerçek `ev.d_rmse_*` dönüşlerini assert etmeden
**yetersizdir**.

```
r_lo_train = 0.50
r_hi_train = 1.50
span_train = max(1.50 - 0.50, 0.1)           # = 1.00
```

Train fikstürü: `split.train` regülatör extrema’sı `0.50` ve `1.50`.
Bu eşitlik **yalnız** ezber hata şeklinin kutu-A oracıdır; formülü
kanıtlamaz. Sabit `range(1.65, 1.85)` L-OVERFIT’i yeşil bırakabilir;
L-BAND-TRAIN öldürür.

Kilitli hata şekli:

```
D_true(r) = 1.0
D_hat(r)  = 1.0 + 2.0 * exp(-((r - 1.75) / 0.05)^2)
r_train_grid = range(0.50, 1.50; length = 80)
```

Gerekçe: ekstra terim train üst kenarında
`2 * exp(-25) ≈ 2.8e-11` (ihmal); dış bant ortasında `2.0`.
Düzgün yumuşak bump.

Zorunlu canlı bağ (onaylı üretim koordinatları):

```
ev = evaluate_holdout(split, evaled, model, params, term, truth_rate)
(R_h, D_h, _) = sample_unknown_destruction_grid(
    model, params, term; r_range = r_holdout, fill_value = 0.3)
(R_d, D_d, _) = sample_unknown_destruction_grid(
    model, params, term; r_range = r_band_external, fill_value = 0.3)
(R_t, D_t, _) = sample_unknown_destruction_grid(
    model, params, term; r_range = r_train_grid, fill_value = 0.3)
d_train = _finite_rate_rel_rmse(D_t, truth_rate(vec(R_t)))
ev.d_rmse_holdout === _finite_rate_rel_rmse(D_h, truth_rate(vec(R_h)))
ev.d_rmse_holdout_domain === _finite_rate_rel_rmse(D_d, truth_rate(vec(R_d)))
d_train < 1e-8
ev.d_rmse_holdout > 0.5
ev.d_rmse_holdout_domain > 0.5
```

`r_holdout` = fiilen gözlenen holdout regülatör koordinatları.
`r_band_external` = kilitli train-türevli dış bant (80 nokta).
Holdout occupancy: iki sentetik deney, regülatör sütunları `1.75`.
Marj platform gürültüsünün üzerindedir (~7 mertebe).

`evaluate_holdout` **kullanmaz**: `normalize_destruction_samples`,
`equation_to_function`, sembolik rekonstrüksiyon,
`sample_learned_function`, train-only / keyfi sabit / tam-set
ızgara.

Öldürür: hatanın train ızgarasında / tam-set ızgarasında /
`range(0.05, 2.0; length=80)` NN ızgarasında (RMSE ≈ 0.35 < 0.5)
ölçülmesi; sembolik rekonstrüksiyon; `normalize_destruction_samples`
ile bump’ın silinmesi; test-içi paralel formülün `ev.d_rmse_*`
yerine geçmesi.

#### L-RES-HOLD — tam falsifikasyon (V1 ile aynı güç)

Fikstür: `ρ_8 != ρ_9`, ikisi sonlu. Test **mutlaka**
`ev = evaluate_holdout(...)` çağırır:

```
ρ_8 != ρ_9
ev.data_residual_holdout === (ρ_8 + ρ_9) / 2
ev.data_residual_train === (ρ_1 + ρ_2 + ρ_3 + ρ_4 + ρ_5 + ρ_6 + ρ_7) / 7
data_residual != data_residual_train
```

Kabul edilmez: RMS, concat, tek IC, medyan, ağırlıklı ortalama.

Zorunlu davranış:

1. `ρ_8 != ρ_9` ⇒ `ev.data_residual_holdout === (ρ_8 + ρ_9) / 2`
2. Yalnız deney 8 gözlemini değiştir ⇒ `ev.data_residual_holdout`
   değişir; legacy `data_residual` değişmez; `ev.data_residual_train`
   değişmez
3. Yalnız deney 1 gözlemini değiştir ⇒ legacy `data_residual`
   değişir; `ev.data_residual_holdout` değişmez
4. Legacy `data_residual` mevcut IC[1] hesabıdır

Bir `ρ_i === Inf` ⇒ ilgili agrega `Inf`.

#### L-GATE — 0.30 holdout kapısı yok (V1 ile aynı güç)

Legacy `data_residual <= 0.30` **değişmez**. Yeni M2 holdout kapısı
**yoktur**. Metin `"0.30"` depo genelinde yasak **değildir**.
`unique_claim_kpis_hold` mevcut M1 fonksiyonudur; holdout okumaz.

Gerçek M2 üretim-yolu fikstürü (UDE eğitimi yok):

```
data_residual_holdout > 0.30
data_residual <= 0.30          # legacy IC[1]
```

Zorunlu assertler:

```
ev = evaluate_holdout(split, evaled, model, params, term, truth_rate)
ev.data_residual_holdout === (ρ_8 + ρ_9) / 2
ev.data_residual_holdout > 0.30
isfinite(ev.data_residual_holdout)
holdout = ev
holdout !== nothing
result = report_recovery(evaled, ident; split, holdout = ev)
result.holdout !== nothing
result.holdout.data_residual_holdout === ev.data_residual_holdout
result.data_residual == evaled.data_residual
result.success == evaled.success
unique_claim_kpis_hold(result.locked_kpis) === true
```

Öldürür:

```
if holdout.data_residual_holdout > 0.30
    holdout = nothing
end
if holdout.data_residual_holdout > 0.30
    evaled = (; evaled..., success = false)
end
HoldoutEvidence(..., Inf, ...)
```

M2 holdout kanıtı gözlemseldir. Yeni M1 kapısı **değildir**.
Yeni holdout eşiği **yoktur**.

#### L-EARLY — karar tablosu

Tam olarak üç durum; A ile B **çökertilmez**:

```
A: training_ok == false
B: training_ok == true && discovery.success == false
C: training_ok == true && discovery.success == true
```

`nothing` **yalnız A** içindir. Neden: `training_ok == false`
(`evaled.discovery === nothing`, M1 kodlaması) iken
`evaluate_holdout` çağrılmaz; Q7 kanıtı yoktur. B ve C
`nothing` **atamaz**.

Case B politikası **kilitlidir**:

- `evaluate_holdout` çağrılır
- `holdout !== nothing`
- Q7 kanıtı durur
- hiçbir holdout metriği **yalnız** sembolik keşif fail olduğu
  için `Inf` yapılmaz

`evaled.success` veya `evaled.discovery.success` örtük Q7
kapısı **değildir**.

A/B/C production kararı **açıktır**. Tek onaylı `holdout = ...`
bloğu (her unique-claim section; ident → `report_recovery`
arası):

```
if evaled.discovery === nothing
    holdout = nothing
else
    holdout = evaluate_holdout(...)
end
```

Holdout kararı olarak **kullanılmaz**:

```
if !evaled.success
if evaled.success == false
if !evaled.discovery.success
if !discovery.success
```

Üç durum **ayrı** test edilir. Tek `success` koşulu A ile B’yi
çökertemez. String yokluğu tek koruma **değildir**.
Struct örneği tek başına yetmez; production karar / rapor
yolu **çalıştırılır**.

A: canlı `ude_adam = 0` yolu. `training_ok == false` →
`holdout === nothing`, `evaluate_holdout` çağrılmaz,
`discovery === nothing`, legacy `data_residual === Inf`.

B (kilitli politika; Q7-görünür):
`training_ok == true && discovery.success == false`
iken Q7 holdout öngörü / mekanistik değerlendirme **açıktır**.
Q5 keşif başarısı örtük Q7 kapısı **değildir**.

Odaklı Case B fikstürü (tam UDE eğitim işi **yok**):

```
evaled_B.discovery !== nothing
evaled_B.discovery.success == false
evaled_B.success == false
# training_ok == true kodlaması: discovery !== nothing
```

Bu fikstür `training_ok == true && discovery.success == false`
durumuna **ulaşır**. Sonra ilgili production karar / rapor yolu
çalıştırılır (kilitli if/else; section ile bit-eşit).
`evaluate_holdout` **çağrılır**. Yalnız
`HoldoutEvidence(...)` kurmak yetmez.

```
if evaled_B.discovery === nothing
    holdout = nothing
else
    holdout = evaluate_holdout(
        split, evaled_B, model, params, term, truth_rate)
end
holdout !== nothing
ev = holdout
ev.data_residual_train, ev.data_residual_holdout,
ev.d_rmse_holdout, ev.d_rmse_holdout_domain
# dördü HoldoutEvidence sözleşmesine göre tanımlı / sonlu
# (fikstür solve’u başarılı; Inf-as-failure yok;
#  Inf yalnız keşif fail diye yazılmaz)
result = report_recovery(evaled_B, ident; split, holdout = ev)
result.holdout !== nothing
```

Aynı `model` / `params` / `split` / `truth_rate` (aynı öğrenilmiş
`D` / aynı holdout girdileri) ile Case C
`evaled_C` (`discovery.success == true`) için:

```
ev_C = evaluate_holdout(split, evaled_C, model, params, term, truth_rate)
ev.data_residual_train === ev_C.data_residual_train
ev.d_rmse_holdout === ev_C.d_rmse_holdout
ev.d_rmse_holdout_domain === ev_C.d_rmse_holdout_domain
ev.data_residual_holdout === ev_C.data_residual_holdout
```

Neden: holdout değerlendirmesi sembolik keşif başarısından
**bağımsızdır**. Keşif nesnesi metrik girdisi değildir.

`evaluate_holdout` + dört M2 yardımcısı gövdesinde `success` /
`discovery.success` kapısı **yoktur**.

C: `training_ok == true && discovery.success == true` →
`holdout !== nothing`, `HoldoutEvidence` doludur; gerçek Q7 kanıtı
üretilir.

Her unique-claim section gövdesinde ident → `report_recovery`
arasında `holdout =` **yalnız** kilitli if/else’dir. Şu token’lar
holdout atama koşulu olarak **yoktur**:

```
evaled.discovery === nothing || evaled.success == false
!discovery.success
!evaled.discovery.success
evaled.success == false
!evaled.success
evaled.discovery.success
```

Öldürür:

```
if !evaled.discovery.success
    holdout = nothing
end
if !evaled.success
    holdout = nothing
end
if evaled.success == false
    holdout = nothing
end
# evaluate_holdout içinde:
if !evaled.discovery.success
    return HoldoutEvidence(Inf, Inf, Inf, Inf)
end
if evaled.success == false
    return HoldoutEvidence(Inf, Inf, Inf, Inf)
end
```

A’yı `Inf` Q7 skalerleri ile kodlamak; `Inf`’i eksik
`HoldoutEvidence` diye kullanmak; `success`’i yeni M2 kapısı
yapmak; yalnız struct kurup production karar yolunu atlamak.

---

### M2-H — hard / benchmark / docs / son denetim

[test/test_recovery_hard.jl](test/test_recovery_hard.jl): legacy
kapılar + `holdout !== nothing` iken sonlu `data_residual_holdout`
ve `d_rmse_holdout`; 0.30 kopyalanmaz; fail ⇒ indeks oynanmaz.

[benchmark/recovery_suite.jl](benchmark/recovery_suite.jl): script
`data_resid=` yanına `holdout_resid=` / `d_rmse_holdout=` eklenebilir;
protokol satırına değil.

Doküman: [docs/src/design/v1_contract.md](docs/src/design/v1_contract.md)
Q7 “not implemented” → “reported, not a gate”; Q4 “not implemented”
kalır. architecture / unique-claim / benchmarks.

Dokunulmayacak kilitler: export, eşik, `PROTOCOL_RESULT_FIELDS`,
Skip `ref_exp = first(...)`, TrainingReuse iğnesi,
`UNIQUE_CLAIM_EXAMPLE_MUST_CONTAIN`, HybridCompose
`sample_unknown_destruction`, `ude_extras_denominator_row`
Recovery.jl’de, `UNIQUE_CLAIM_PROTOCOL` alan listesi,
`examples/unknown_inhibition.jl` suite’e bağlamak zorunda değil.

---

## Sızıntı kataloğu

| Yanlış gövde | Test |
|---|---|
| Yanlış 7 IC / değer-eşit kopya | L-SPLIT-ID |
| Holdout `Experiment(...)` / değer-eşit kopya (splitter) | L-SPLIT-ID |
| `return fit, generate_recovery_experiments(...)` / ikinci generate / gizli `_regen` / section-split-eval generate | L-RNG |
| `pop!` / `insert!` / `resize!` / `splice!` / restore; `_carve_and_restore!` / `_split_impl` / `_prepare!` / `_partition!` / `_prepare_holdout` / `_temporary_partition` dolayımı | L-SET-INTACT |
| Train/holdout `set.metadata` içinde | L-SET-META, L-SPLIT-META |
| `fit_unknown_destruction(..., set)` | L-FIT-A |
| `fit(split.train)` sonra `_polish_full(..., set, ...)` | L-FIT-A |
| `fit(split.train)` sonra `train_experiments_with_warmup(..., set, ...)` | L-FIT-A |
| Y2-SUITE: `_train_unknown_edge` sonrası `train_experiments_with_warmup(ude_fit.params, ude_set, ...)` | L-FIT-A |
| Unique-claim section’da `train_ude(` / ikinci `fit_unknown_destruction(` | L-FIT-A |
| `_train_unknown_edge` içi gizli ikinci fit | L-FIT-A |
| `length==7` ama IC 2–8 | L-FIT-B |
| `_regulator_grid(ude_set, term)` / `_regulator_grid(set, term)` / holdout union | L-DOM-A, L-DOM-B |
| `r_range` overwrite / `union` / holdout extrema / inline tam-set | L-DOM-A, L-DOM-B |
| `_peek_holdout!` / `discover_equations(...holdout...)` | L-DISC-A |
| `evaluate_holdout` → `_peek_holdout` → `discover_equations` | L-DISC-A |
| `evaluate_holdout` → `helper` → `discover_unknown_destruction` | L-DISC-A |
| `_evaluate_unknown_rate_recovery(...; holdout = split.holdout)` / `split = split` | L-DISC-B-1 |
| `data_residual_fn = d_hat -> something_using(split.holdout)` | L-DISC-B-1 |
| Composer → `_composer_helper` → `discover_unknown_rate(... split.holdout.observations ...)` | L-DISC-B-2 |
| Composer → `_helper` → `discover_equations(holdout.times, holdout.derivatives, ...)` | L-DISC-B-2 |
| Composer → `helper` → `discover_unknown_destruction(...)` | L-DISC-B-2 |
| Composer keşfine holdout `times` / türev / occupancy (doğrudan veya yardımcı) | L-DISC-B-2, L-DISC-B-3 |
| Sabit dış bant / holdout extrema bandı | L-BAND |
| Holdout \(D\) yanlış ızgara / sembolik / normalize | L-D-OCC, L-OVERFIT |
| Test `ev.*` okumadan metrik hesaplar | L-D-OCC, L-OVERFIT |
| `data_residual = data_residual_train` | L-RES-LEGACY |
| Holdout residual IC[1] / RMS / concat / medyan / ağırlıklı ortalama | L-RES-HOLD |
| Test `ev.data_residual_holdout === (ρ_8 + ρ_9) / 2` okumaz | L-RES-HOLD |
| `success == false` / `!discovery.success` ⇒ `holdout = nothing` | L-EARLY |
| `if !evaled.success; holdout = nothing` | L-EARLY |
| `if !evaled.discovery.success; return HoldoutEvidence(Inf, Inf, Inf, Inf)` | L-EARLY |
| Case B’de `holdout = nothing` / `HoldoutEvidence(... Inf ...)` | L-EARLY |
| `data_residual_holdout > 0.30` ⇒ `holdout = nothing` / `success = false` / `Inf` | L-GATE |
| `report_recovery` → `_ensure_holdout` → `evaluate_holdout` | L-SITE |
| Composer / rapor / kapanış içi `evaluate_holdout` | L-SITE |
| Public export / kilit listesi şişirme | L-API |
| M3/M4 alanı | L-FIELDS, L-M34 |

Uzunluk-yalnız testler yetersizdir.
L-RNG splitter birim testi **değildir**; observation `==`
yetmez. Depo-geneli `generate_*` / RNG yasağı **yoktur**.
Genel kaynak-string honesty envanteri büyütülmez.

## Public API testi

```
for name in (:ExperimentSplit, :HoldoutEvidence, :evaluate_holdout,
             :unique_claim_experiment_split,
             :UNIQUE_CLAIM_TRAIN_INDICES, :UNIQUE_CLAIM_HOLDOUT_INDICES,
             :_holdout_observed_regulators,
             :_unique_claim_external_regulator_band,
             :_finite_rate_rel_rmse, :_mean_hybrid_residual)
    !(name in names(BioDynaX))
    !(name in LOCKED_PUBLIC_EXPORTS)
end
LOCKED_PUBLIC_EXPORTS bit-eşit pre-M2
```

Kilidi genişletmek çözüm değildir.

## Bayat terimler (yalnız yasak)

Aşağıdakiler onaylı uygulama talimatı **değildir**. Aktif
satırlarda yalnız **WRONG IMPLEMENTATION** / yasak / “değildir”
olarak dururlar. V1 master hijyeni ile aynı kural.

| Terim | Bu belgede | Durum |
|---|---|---|
| `split_experiments` | Belge otoritesi; Yeni soyutlamalar; M2-A yasak; Kesinlikle M2 dışında | **WRONG** — yasak genel splitter adı; onaylı üretici `unique_claim_experiment_split` |
| 6/3 split | Belge otoritesi; M2-A yasak | **WRONG** — yasak yanlış indeks; onaylı 7/2 |
| `equation_to_function` holdout \(D\) | Residual; Holdout D yasağı; L-DISC-A; L-D-OCC; L-OVERFIT; Bayat tablo | **WRONG** — yasak holdout \(D\) yolu |
| `normalize_destruction_samples` holdout \(D\) | Holdout D yasağı; L-DISC-A; L-D-OCC; L-OVERFIT; Bayat tablo | **WRONG** — yasak holdout \(D\) yolu |
| `unique_claim_kpis_hold` holdout kapısı | L-GATE; Kesinlikle M2 dışında; Bayat tablo | **WRONG** — holdout kapısı olarak yasak |
| `unique_claim_kpis_hold` M1 IC[1] | L-GATE (`=== true` legacy kapı korunur) | **ACTIVE M1** — holdout okumaz; silinmez |
| bağımsız holdout generate | L-RNG Y-RNG-*; Bayat tablo | **WRONG** — dokuz deney tek generate, sonra split |
| birden fazla `evaluate_holdout` sahibi | Tek üretim çağrı yeri; L-SITE | **WRONG** — tek sahip, iki unique-claim kopyası |
| composer içinde holdout keşfi | L-DISC-B-1/2/3 | **WRONG** — M1 composer train-grid + dummy time; holdout keşif girdisi değil; geçişli yardımcı grafı dahil |
| `rate_rel_rmse` holdout arayüzü | Holdout D | **WRONG** — test `ev.d_rmse_*` okur |
| sembolik \(D\) rekonstrüksiyonu | Holdout D; L-OVERFIT | **WRONG** |

## Kabul

V1 kabul maddeleri 1–18 ve tüm L-* testleri birlikte yeşil olmadan
M2 kabul edilmez. `HoldoutEvidence` struct’ı tek başına kabul
değildir. L-RNG kaynak `count == 1` + davranışsal provenance
birlikte zorunludur; splitter-only L-RNG yeşil **bırakmaz**.

## Rollback

`fit_unknown_destruction`’a tam 9-IC seti geri ver.
`_regulator_grid(ude_set, term)` tam sete dönsün.
`ExperimentSplit` / `HoldoutEvidence` / `evaluate_holdout` /
`unique_claim_experiment_split` / dört M2 yardımcısı kalksın.
MRR `split` / `holdout` alanlarını bıraksın.
M1 “`ExperimentSplit` yok” kilitleri geri takılsın.
`data_residual` adı durur.

## Kesinlikle M2 dışında

M3 fonksiyonel identifiability; M4 yörünge-örnekli keşif;
holdout Fisher; uncertainty; hypothesis; `ExperimentSet` şişirme;
public splitter; `split_experiments`; ikinci pipeline; 0.30 holdout
kapısı; `unique_claim_kpis_hold` holdout kapısı; `LOCKED_PUBLIC_EXPORTS`
genişletme; `data_residual` rename.

## Bu görevde yapılmayacaklar

Kaynak, test, benchmark veya diğer markdown dosyalarına yazılmaz.
M2 kodu uygulanmaz. M0 / M1 / M3–M10 bilimsel içeriği değişmez.
