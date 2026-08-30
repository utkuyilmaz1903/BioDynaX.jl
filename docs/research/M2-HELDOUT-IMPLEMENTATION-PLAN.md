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
RMS yoktur. Concat yoktur. Aritmetik ortalama.

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

Holdout \(D\) testleri **her zaman**:

```
ev = evaluate_holdout(...)
ev.d_rmse_holdout
ev.d_rmse_holdout_domain
```

Bağımsız yeniden hesap `ev.*` yerine geçemez. Üretim koordinatları
`evaluate_holdout` sırasında `sample_unknown_destruction_grid`’e
fiilen geçen `r_range`’dir.

`evaluate_holdout` gövdesinde yok:

- `normalize_destruction_samples(`  (yasak holdout \(D\) yolu)
- `equation_to_function(`           (yasak holdout \(D\) yolu)
- `sample_learned_function(`
- sembolik `D` rekonstrüksiyonu
- `discover_equations(`
- `discover_unknown_rate(`
- `discover_unknown(`
- `_peek_holdout`
- keyfi NN ızgarası / onaylı protokolle ilgisiz sabit ızgara
- holdout \(D\) için train-only ızgara
- `evaled.success` / `discovery.success` kapısı

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

## Erken durum A / B / C

Q7 yokluğunun tek temsili: `holdout === nothing`.

| | A | B | C |
|---|---|---|---|
| `training_ok` | `false` | `true` | `true` |
| `discovery` | `=== nothing` | `!== nothing`, `success == false` | `!== nothing`, `success == true` |
| `evaluate_holdout` | çağrılmaz | **çağrılır** | **çağrılır** |
| `holdout` | `=== nothing` | `HoldoutEvidence` (`!== nothing`) | `HoldoutEvidence` (`!== nothing`) |
| Q7 alanları | yok | HoldoutEvidence sözleşmesine göre sonlu / tanımlı | aynı |
| Sembolik keşif | gerekmez / yok | **gerekmez** | vardır; Q7 girdisi değildir |
| legacy `data_residual` | `Inf` | `evaluate_recovery` (M1) | IC[1] hybrid |

Case B kilitli politikadır: Q7 holdout öngörü / mekanistik
değerlendirme, sembolik keşif başarısız olsa da **açıktır**.
Q5 keşif başarısı örtük Q7 kapısı **değildir**.

Aşağıdaki gövde yasaktır:

```
if !evaled.discovery.success
    holdout = nothing
end

if evaled.success == false
    holdout = nothing
end

if !evaled.success
    holdout = nothing
end
```

Tek karar kuralı:

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

Unique-claim için **tam bir** fitting işlemi:

```
fit_unknown_destruction(..., split.train)
```

`fit_unknown_destruction` içindeki
`train_experiments_with_warmup(..., split.train, ...)` bu tek
işlemin parçasıdır.

Sonra: tam-set eğitim yok; `train_experiments(` yok;
`train_experiments_with_warmup(..., set, ...)` yok; eşdeğer gizli
eğitmen yok; `_polish_full(..., set, ...)` yok; dokuz deney
üzerinde ikinci optimizer geçişi yok.

Bu, depo-geneli `train_experiments*` yokluğu **değildir**.
`:partial_obs`, `Training.jl`, `TrainingReuse.jl`,
`ExperimentCheckpoint.jl` meşru çağrılar içerir.
`count("train_experiments_with_warmup", repo) == 0` **yanlış
sözleşmedir**.

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

Yanlış gövde kırmızı (L-FIT-A):

```
fit_unknown_destruction(..., split.train)
_polish_full(..., set, ...)

fit_unknown_destruction(..., split.train)
train_experiments_with_warmup(..., set, ...)
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

`length(split.train) == 7` yetmez.

Test: L-FIT-A (dar kaynak + davranışsal sentinel), L-FIT-B,
L-RNG (kaynak `count == 1` + davranışsal provenance).

---

### M2-C — train-derived regulator domain

Dosya: [src/Recovery.jl](src/Recovery.jl) `:ude_discovery` ve
`:mm_unknown`. Composer imzası durur. `ExperimentSplit` almaz.

```
r_range = _regulator_grid(split.train, term)
```

`data_residual_fn` hâlâ `ref_exp = first(ude_set.experiments)` (IC[1]).

Yanlış: `_regulator_grid(ude_set, term)`; holdout extrema `union`;
token sonrası overwrite.

Test: L-DOM-A, L-DOM-B. Sentinel fikstür zorunlu.

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

Gövde yasağı (L-DISC-A, L-D-OCC):

- `discover_unknown_rate(`
- `discover_unknown(`
- `discover_equations(`
- `_peek_holdout`
- `normalize_destruction_samples(`
- `equation_to_function(`
- `sample_learned_function(`
- `evaluate_recovery(`
- `_ensure_holdout`
- holdout \(D\) için train-only / keyfi sabit ızgara
- `success` / `discovery.success` kapısı
- holdout metriği `> 0.30` kapısı

RecoveryPipeline.jl:

```
count("discover_equations(", file) == 0
count("discover_unknown_rate(", file) == 0
count("_peek_holdout", file) == 0
count("evaluate_holdout(", file) == 1
```

Canlı test **her zaman** `ev = evaluate_holdout(...)` çağırır ve
`ev.d_rmse_holdout` / `ev.d_rmse_holdout_domain` /
`ev.data_residual_holdout` okur.
Bağımsız `_finite_rate_rel_rmse(...)` `ev.*` yerine geçemez.

L-SET-INTACT: `evaluate_holdout` ikinci giriş noktasıdır. Ondan
erişilebilir yerel yardımcılar (`_prepare_holdout`,
`_temporary_partition` dahil) orijinal `ExperimentSet` /
`set.experiments` üzerinde yasaklı mutator kullanamaz.
Giriş-noktası-yalnız gövde taraması **yetersizdir**. Depo-geneli
mutator yasağı **yoktur**.

Fonksiyon tanımı M2-D’dedir. Tek üretim çağrı yeri M2-F’de
kablolanır. M2-D alternatif sahip açmaz. Birim test doğrudan
fonksiyonu çağırır; production sıra sözleşmesi L-SITE’tir.

Test: L-DISC-A, L-DISC-B, L-BAND, L-D-OCC, L-RES-HOLD, L-OVERFIT,
L-EARLY (Case B), L-GATE.

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
eğitimsiz). L-DISC-A, L-DISC-B, L-BAND, L-D-OCC, L-OVERFIT,
L-RES-LEGACY, L-RES-HOLD, L-EARLY, L-GATE, L-SET-INTACT, L-FIT-A,
L-RNG.

L-RNG **mutlaka** production unique-claim yolunu enstrümante
eder veya çalıştırır (`_train_unknown_edge` + suite split +
`evaluate_holdout`). Splitter birim testi L-RNG **değildir**.
`ids_gen[i] ===` eğitim ve holdout değerlendirme nesneleri;
`n_gen == 1`. `set2` gözlem-eşit ikinci generate kırmızı kalır.

L-OVERFIT **mutlaka**:

```
ev = evaluate_holdout(...)
d_train < 1e-8
ev.d_rmse_holdout > 0.5
ev.d_rmse_holdout_domain > 0.5
```

Kutu-A `range(1.65, 1.85)` ezber oracıdır; formülü kanıtlamaz.
L-BAND-TRAIN ikinci kutu `(1.0, 3.0) → range(3.3, 3.7; length=80)`
ile sabit bandı öldürür.

L-DISC-A composer’ın iki `discover_unknown_rate` çağrısı dururken
`_peek_holdout!` / `discover_equations(...holdout...)` kırmızı olur.

L-GATE: üretim `evaluate_holdout` ile
`data_residual_holdout > 0.30`; `holdout !== nothing`;
legacy IC[1] kapısı değişmez. Sentetik `HoldoutEvidence`
enjeksiyonu tek koruma **değildir**.
`unique_claim_kpis_hold` mevcut M1 fonksiyonudur; holdout kapısı
**değildir**.

L-EARLY Case B: `training_ok == true`, `discovery.success == false`,
tam UDE işi yok; üretim karar yolu; `evaluate_holdout` çağrılır;
dört Q7 alanı tanımlı / sonlu.

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
| `length==7` ama IC 2–8 | L-FIT-B |
| `_regulator_grid(ude_set, term)` / holdout union | L-DOM-A, L-DOM-B |
| `_peek_holdout!` / `discover_equations(...holdout...)` | L-DISC-A |
| Sabit dış bant / holdout extrema bandı | L-BAND |
| Holdout \(D\) yanlış ızgara / sembolik / normalize | L-D-OCC, L-OVERFIT |
| Test `ev.*` okumadan metrik hesaplar | L-D-OCC, L-OVERFIT |
| `data_residual = data_residual_train` | L-RES-LEGACY |
| Holdout residual IC[1] / RMS / concat | L-RES-HOLD |
| `success == false` / `!discovery.success` ⇒ `holdout = nothing` | L-EARLY |
| `data_residual_holdout > 0.30` ⇒ `holdout = nothing` / `Inf` | L-GATE |
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

Aşağıdakiler onaylı talimat **değildir**. Yalnızca yanlış
uygulama olarak anılır:

| Terim | Durum |
|---|---|
| `split_experiments` | yasak genel splitter adı |
| 6/3 split | yasak yanlış indeks |
| `equation_to_function` holdout \(D\) | yasak yol |
| `normalize_destruction_samples` holdout değerlendirme | yasak yol |
| `rate_rel_rmse` holdout arayüzü | yasak ad; test `ev.d_rmse_*` okur |
| bağımsız holdout generate | yasak; L-RNG (production path) |
| birden fazla `evaluate_holdout` sahibi | yasak |
| `unique_claim_kpis_hold` holdout kapısı | yasak; M1 IC[1] kapısı durur |
| sembolik \(D\) rekonstrüksiyonu | yasak |

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
