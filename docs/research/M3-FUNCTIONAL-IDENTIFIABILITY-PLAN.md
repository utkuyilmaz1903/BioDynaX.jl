---
name: M3 Functional Identifiability Plan
overview: "Canlı adversarial sözleşme: her kritik Q4 değişmezi assess_functional_identifiability yürütme yolunda gözlenir. Yardımcı doğruluğu veya öz-bildirim yeşil bırakamaz. Kaynak bu belgede uygulanmaz."
todos:
  - id: m3-functional-id-impl
    content: "M3-A…H: LIVE p0+params+D+X; holdout inclusion/HP sentinel; derive-live A/B/C; zero-live; walk; assess CALL; M2 hash; (201…205); LS 37/101; Fisher asymptotic"
    status: pending
isProject: false
---

# M3 — Pratik fonksiyonel identifiability uygulama planı

Bu belge
[docs/research/V1-IMPLEMENTATION-PLAN.md](docs/research/V1-IMPLEMENTATION-PLAN.md)
Milestone 3’ün **uygulama dilimidir**. Çelişide V1 bilimsel kimliği
kazanır; bu belge V1 M3 bölümünü operasyonel olarak kilitler.

Bu metin önceki taslakların yama özeti değildir. **Tam, kendi
içinde tutarlı M3 planıdır.** Adversarial denetim mimariyi kabul
etti; yazılı test sözleşmesi hâlâ bilimsel olarak yanlış bir
implementasyonu yeşil bırakabildiği için **canlı adversarial
sözleşme** olarak kilitlendi.

**Durum:** yalnızca plan. Bu turda kaynak, test, benchmark, CI veya
public API uygulanmaz.

**Otorite sırası**

1. [docs/src/design/v1_contract.md](docs/src/design/v1_contract.md)
   (Q-katmanları, kapalı iddialar)
2. V1 Milestone 3 + bu belge (uygulama kilidi)
3. Mevcut M2 kod ve testler (davranış yüzeyi)

Vizyon belgesindeki “çoklu IC / pertürbasyon” cümlesi **Q7’dir**.
M3, v1 sözleşmesindeki **Q4**’ü uygular.

M2 protokolünü **uzatır**, **yerine geçmez**.

```
generate_recovery_experiments   (tek 9-IC set; zorunlu kwargs)
    ↓
unique_claim_experiment_split   (kilitli 7/2)
    ↓
assess_functional_identifiability   ← tek Q4 sahibi
    domain.z bir kez, restart’tan önce
    5 taze build_ude_model + 5 fit(split.train)
    sample_unknown_destruction_grid(; r_range = domain.z)
    predict_ude (fırlatma yakalanır, yalıtılır; dönen X kaydedilir)
    pairwise D + pairwise x(t) on domain.z
    ↓
FunctionalIdentifiabilityDiagnostic   (kapı değil; bayraklar türetilir)
```

Q3 (`production_destruction_tradeoff` / `unidentifiable_edge`) aynı
kalır. Q7 (`evaluate_holdout` / `HoldoutEvidence`) aynı kalır.
`unique_claim_kpis_hold` Q4 okumaz.

---

## 1. M2-G1 bilimsel test standardı (L-STRENGTH)

M3, M2-G1 saldırı testlerinin bilimsel barını kullanır.

Bilimsel olarak kritik bir değişmez, bir yardımcı fonksiyonun
doğru olmasından **kanıtlanmaz**.

Kritik her değişmez için **dördü birden** zorunludur:

1. **Canlı fonksiyon, çağrılan fonksiyonun içinden** gözlenir.
2. Test beklenen değeri **bağımsız** hesaplar.
3. Test canlı sonucu o bağımsız beklenti ile **karşılaştırır**.
4. En az bir **somut yanlış implementasyon** kırmızıya düşer.

Yetersiz kanıt (kritik değişmez için kabul edilmez):

- yalnız yardımcı (`scale_align_destruction`, constructor,
  `assemble_functional_identifiability_diagnostic`) testi
- yorum / docstring varlığı
- string-presence (`occursin("assess_functional_identifiability", src)`)
  tek başına — özellikle benchmark CALL ve canlı `r_range` için
- öz-bildirimli Q4 metadata
  (`construction = :train_obs_union_holdout_obs`,
  `seed = 201` etiketi, `nn_init_fingerprint` alanına assess’in
  kendi yazdığı sayı, casusun “ben bu tohumu kullandım” iddiası)
- `isdefined` tabanlı yokluk iddiası (M2 Q4 yokluğu için
  `∉ names` zorunludur; `!isdefined` zayıftır)
- `===` nesne kimliği tek başına ( `deepcopy` yeni referans,
  aynı sayılar üretir)

Davranışsal testin gerçekten imkânsız olduğu tek durumda kaynak
testi + gerekçe yazılır. “Yardımcı doğru, o hâlde canlı yol da
doğrudur” gerekçe **değildir**.

**Hazır değil kuralı.** Bir saldırı, yardımcı doğru iken
`assess_functional_identifiability` o yardımcıyı atlayıp veya
sonucu uydurarak yeşil kalabiliyorsa plan hazır **değildir**.

§39’daki T-B-P0, T-B-PARAMS, T-B-HP-SENTINEL, T-B-INC-HOLD,
T-B-PRED203, T-B-ZLIVE, T-C-DBIND, T-C-DSOURCE, T-C-TBIND,
T-C-DERIVE-LIVE, T-C-ZERO-LIVE, T-E-WALK, T-G-CALL, T-E-M2HASH
testleri **LIVE-path** testleridir. Yardımcı testi değildir.

---

## 2. Denetimin bilimsel nedeni

Mimari kabul edildi (kardeş tanı, 7/2, LS şekil, kapı yok).
Yazılı testler yine de yanlış implementasyonu yeşil
bırakabiliyordu, çünkü kritik kilitler yardımcıda, kurucuda
veya öz-bildirimde duruyordu:

| Zayıf kanıt | Yeşil bırakabildiği yanlış |
|---|---|
| Yalnız `p0` / init izi | bir gerçek fit; final params beş kez kopya/`deepcopy`; etiket 201..205 |
| Fit casusu yalnız `ExperimentSet` | beş etiket, tek sampled params; paylaşılan params nesnesi |
| Sample casusu yalnız `r_range` | sampler farklı fit sonuçlarını yutmaz; `z` bağımsız kurulmaz |
| `predict_ude` casusu yalnız girdi params | uydurma yörünge metriği; truth’tan kopya; sabit sayı |
| `function_disagree` kurucu/assemble testi | canlı `assess` bayrağı uydurur; hücre B gizlenir |
| `D_j=0` yalnız yardımcı | canlı çift `eps` / `α=1` / çifti düşer |
| HP yasağı yalnız `for adam in` yokluğu | gizli yardımcı holdout ile tarar veya dahil eder |
| Holdout “HP tarama yok” | holdout, restart dahil/dışını seçer |
| Benchmark string `contains` | yorum, kullanılmayan referans, sahte tanı |
| Keşif yasağı yalnız dosya token’ı | gizli yardımcı `discover_*` çağırır |
| M2 `!isdefined(...)` | tip tanımlanır tanımlanmaz kırılır veya zayıflatılır |

Bu belge her satırı M2-G1 barına ve **canlı `assess` yoluna**
çıkarır.

---

## 3. Mevcut depo gerçekleri (kilit; varsayım değil)

Implementasyon bu gerçeklere uymak zorundadır. “Teorik olarak
makul” imza yeterli değildir.

1. `sample_unknown_destruction_grid(model, p, term; r_range, fill_value)`
   `(R, D, term)` döner. `D` **1×N Matrix**’tir. Q4 vektörü
   `D = vec(D_matrix)` ile alınır.
   `length(D) == length(domain.z)` zorunludur.
   Varsayılan `r_range = range(0.05, 2.0; length = 80)`’dir.
   Bu varsayılan Q4 domain’i **değildir**.

2. `predict_ude` durum × zaman matrisi döner
   (`nstates × ntimes`). Başarısız solve’da **fırlatır**;
   non-finite dizi vaat etmez. Her restart `try/catch` ile
   yalıtılır. Girdi params kaydı, dönen `X` kaydının yerine
   geçmez.

3. `generate_recovery_experiments(rng, truth_net, truth_params;
   tspan, n_points, noise_σ, …)` zorunlu keyword’ler
   `tspan`, `n_points`, `noise_σ`’dır. `protocol=` geçerli
   değildir. Veri üretimi (benchmark / nightly) bunları
   `UNIQUE_CLAIM_PROTOCOL` alanlarından açar:
   `tspan=(0.0, 8.0)`, `n_points=50`,
   `noise_σ = UNIQUE_CLAIM_PROTOCOL.observation_noise` (`0.0`).

4. `term = only_unknown_destruction(model)`. Occupancy ve grid
   `term.regulator` kullanır.

5. M2 / unique-claim protokolü:

   ```
   adam_iterations = 100
   bfgs_iterations = 50
   tspan = (0.0, 8.0)
   n_points = 50
   seed = 103
   ```

   `UNIQUE_CLAIM_PROTOCOL` bu değerleri taşır.
   `RECOVERY_THRESHOLDS` Q4 okumaz ve Q4 tarafından okunmaz.

6. Restart bağımsızlığı **model init**’tedir:
   `build_ude_model(MersenneTwister(seed), ude_net)`.
   `fit_unknown_destruction` imzasında bugün `seed` **yoktur**.
   Fit `seed` anahtarı (eklense bile) bağımsızlık kanıtı
   **değildir**. M3 restart bağımsızlığı gerçek
   `build_ude_model` başlatmasından gelir; bir fit keyword’ünden
   değil.

7. `fit_unknown_destruction(ude_model, ude_p0, set; adam, bfgs,
   frozen_phys, phys_init)` tek onaylı eğitmendir.
   Mevcut `FIT_UNKNOWN_DESTRUCTION_OBSERVER` **yalnız**
   `ExperimentSet` görür (`_note_fit_unknown_destruction(set)`).
   `p0`, tohum ve eğitim kwargs’ını **görmez**. M2 sözleşmesi
   `observer(set) -> Union{TrainingResult,Nothing}` olarak
   korunur. M3, bu M2 casusunun üzerine **ayrı bir fit-giriş
   gözlem yolu** ekler (L-FITSPY). Fit-giriş casusu eğitimi
   yutmaz.

8. Mevcut `SAMPLE_UNKNOWN_DESTRUCTION_GRID_OBSERVER` gerçek
   `r_range` görür; dönen `(R, D, term)`’i ve `p` argümanını
   **görmez**. M2 `observer(r_range)` korunur. M3, canlı `D`
   ve canlı sampled-params bağlamak için içeriden sonuç +
   params gözlemi ekler (L-SAMPSPY, T-B-PARAMS, T-C-DBIND).

9. `predict_ude` içinde bugün casus **yoktur**. M3, canlı
   `predict_ude` için M3-yerel çalışma-anı gözlem yolu ister
   (L-PREDSPY). Casus **dönen `X`**’i kaydeder. Yalnız girdi
   `params` kaydı yörünge bağını kanıtlamaz (T-C-TBIND).

10. `ExperimentSplit` 7/2 görünümüdür; `train.experiments` ve
    `holdout.experiments` orijinal `Experiment` nesneleridir.
    Holdout eğitime girmez. Q4 domain ve fit,
    `split.train` / `split.holdout` **deney listesini** okur;
    indeksten yeniden üretmez.

11. `HoldoutEvidence` dört skalerdir; Q4 değildir.

12. `MechanismRecoveryResult` alanında
    `functional_identifiability` **yoktur** ve eklenmez.

13. `TrainingRetcode` ∈ `{Success, NotConverged, BFGSFailure,
    GradientFailure, ODEFailure}`. `NotConverged` başarı
    filtresi değildir.

14. Julia’da unexported tip için
    `isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)`
    **true** olur. Mevcut M2 `!isdefined(...)` iddiaları tip
    tanımlanır tanımlanmaz kırılır. Retarget
    `FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)`
    zorunludur. `!isdefined` ile değiştirmek M2’yi zayıflatır
    ve yasaktır.

15. `rate_rel_rmse(estimate, truth)` vektörleştirir; ölçek
    ikinci argümandır:
    `max(sqrt(mean(abs2, truth_vec)), eps(Float64))`.

16. L-FIT-A: `_train_unknown_edge` içinde
    `fit_unknown_destruction(` sayısı **1** kalır. M3 çağrıları
    yalnız `FunctionalIdentifiability.jl` içindedir.
    `Recovery.jl`’e Q4 çağrısı eklemek M2’yi kırar.

17. `include` yeri: `RecoveryPipeline.jl` sonrası
    (`src/BioDynaX.jl`). `ExperimentSplit`,
    `fit_unknown_destruction`, `sample_unknown_destruction_grid`,
    `UNIQUE_CLAIM_PROTOCOL` o anda görünür.

---

## 4. Bilimsel tanım

**Soru:** Aynı sabit M2 train/holdout verisinde, bağımsız UDE
fit’leri gözlenen yörüngelerde anlaşırken yıkım fonksiyonları
\(\hat D_i(z)\) pratik olarak farklı olabilir mi?

Bu, **pratik fonksiyonel-identifiability tanısıdır**.

| Bu tanı değildir | Bu tanı ölçer |
|---|---|
| Yapısal identifiability sertifikası | Bağımsız \(\hat D_i\) vs \(\hat D_j\) (ortak `z`, LS ölçek-normalize) |
| Bayes / credible interval | Bağımsız \(\hat x_i(t)\) vs \(\hat x_j(t)\) (aynı IC’ler) |
| Q3 ölçek / Fisher uyarısı | `trajectory_agree_function_disagree` etiket bayrağı |
| Q1/Q2 (veriye / truth’a fit) | Beş restart envanteri (başarı + başarısızlık) |
| Q5 sembolik destek | — |
| Q7 held-out kapısı | — |
| Hipotez sıralama / M4 occupancy keşif | — |

Tek fit Q4 üretmez. “Yörünge uydu ⇒ \(D\) tek” iddiası ancak bu
tanıyla sınanır.

**Bilimsel M3 sonucu:** unknown Hill
(`admit_recovery_suite_network(:ude_discovery)`).
MM aynı fonksiyonlarla `family=:mm` olabilir; ikinci ürün iddiası
değildir. Hard MM kapıları değişmez.

---

## 5. Korunacak M2 değişmezleri (L-M2INV)

M3 bunları değiştirmez:

- kilitli 7/2 split (`UNIQUE_CLAIM_TRAIN_INDICES` /
  `UNIQUE_CLAIM_HOLDOUT_INDICES`)
- tek `generate_recovery_experiments` (unique-claim production)
- train-only M2 fit (`fit_unknown_destruction(..., split.train)`)
- train-türevli M2 keşif domain’i (`_regulator_grid` ICs 1..7)
- holdout \(D\) gözlenen koordinatta, **gerçek nöral** \(\hat D\)
- train-türevli dış bant (`_unique_claim_external_regulator_band`)
- aritmetik residual toplama (7’nin / 2’nin ortalaması; RMS değil)
- legacy IC[1] `data_residual`
- Q7 raporlanır, kapı değildir
- M2 `0.30` kapısı semantiği değişmez; holdout’a kopyalanmaz
- `evaluate_holdout` sahipliği ve sırası değişmez
  (`ident → evaluate_holdout → report_recovery`)
- M2 public API değişmez
- `RECOVERY_THRESHOLDS` değerleri değişmez
- `LOCKED_PUBLIC_EXPORTS` değişmez
- `UNIQUE_CLAIM_KPI_FIELDS` / `PROTOCOL_RESULT_FIELDS` değişmez
- `unique_claim_kpis_hold` Q4 okumaz
- `count("evaluate_holdout(", "src/Recovery.jl") == 2`
- `functional_identifiability ∉ fieldnames(MechanismRecoveryResult)`
- `ExperimentSet` train/holdout sahibi değildir
- Case A/B/C holdout semantiği durur
- unique-claim stdout blokları değişmez

`functional_identifiability ∉ fieldnames(MechanismRecoveryResult)`
**korunur**. Q4 tanısı `MechanismRecoveryResult`’a eklenmez.
Zayıflatılmaz.

---

## 6. M2 test bütünlüğü (L-M2INT, T-E-M2HASH)

M3, M2 testlerini silmez, gevşetmez, `isdefined` tabanlı daha
zayıf iddialarla değiştirmez.

**Korunması zorunlu M2 kilitleri / test aileleri:**

- M2-G1 saldırı testleri
- M2-G2 hard testler
- L-SPLIT-ID
- L-SET-INTACT
- L-FIT-A / L-FIT-B
- L-RNG
- L-DOM-A / L-DOM-B
- L-D-OCC
- L-OVERFIT
- L-RES-HOLD / L-RES-LEGACY
- L-DISC-A / L-DISC-B-1 / L-DISC-B-2 / L-DISC-B-3
- L-EARLY
- L-GATE
- L-SITE
- L-API
- L-M34

Tek izinli M2 retarget (Q4 tipi artık var olacağı için):

```
# YASAK (zayıf; unexported tipte true olur)
!isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)

# ZORUNLU
FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
```

Bu retarget şuralarda yapılır ve çevresindeki M2 assert’ler durur:

- `test/test_holdout.jl` (mevcut 1791, 2185, 3069)
- `test/test_recovery_pipeline.jl` (mevcut 576, 723)

Ayrıca **korunur**:

```
functional_identifiability ∉ fieldnames(MechanismRecoveryResult)
```

Q4 tanısı `MechanismRecoveryResult`’a eklenmez.

`BioDynaX.jl` `include("FunctionalIdentifiability.jl")` bu
`∉ names` retarget’i **olmadan** yapılamaz.

T-E-M2HASH (§39) bu listenin canlı test dosyalarında durduğunu
kanıtlar. Yorum (“M2 testleri korunur”) yeterli değildir.

---

## 7. Tek Q4 sahibi (L-OWNER)

Yalnız

```
assess_functional_identifiability
```

şunların sahibidir:

- domain inşası
- restart döngüsü
- taze model init
- eğitim
- dahil etme elverişliliği (M3 kriterleri; holdout kalitesi değil)
- \(D\) örnekleme
- yörünge tahmini
- çift inşası
- tanı montajı

Sahibi **değildir:**

- Q1 (hybrid residual vs data; IC[1] kapısı)
- Q3 (`production_destruction_tradeoff`)
- Q5 (`discover_unknown_rate` / `discover_equations`)
- Q7 (`evaluate_holdout`)
- M2 orkestrasyon (`run_recovery_suite`, `_train_unknown_edge`)
- `report_recovery`

İç yardımcı (`scale_align_destruction`,
`pairwise_destruction_metrics`,
`fit_functional_identifiability_restart`) yalnızca `assess`
tarafından çağrılabilir. Yardımcı testleri canlı yolun yerine
geçmez (L-STRENGTH).

`assess` şuralarda **yoktur** (kaynak kilidi):

```
count("assess_functional_identifiability(", "src/Recovery.jl") == 0
count("assess_functional_identifiability(", "src/RecoveryPipeline.jl") == 0
count("evaluate_holdout(", "src/FunctionalIdentifiability.jl") == 0
count("report_recovery(", "src/FunctionalIdentifiability.jl") == 0
count("run_recovery_suite(", "src/FunctionalIdentifiability.jl") == 0
```

Kaynak sayacı T-E-WALK’ın yerine geçmez. Gizli yardımcı kaçış
değildir.

---

## 8. Kilitli sabitler

```julia
const FUNCTIONAL_ID_DATA_SEED = 103

const FUNCTIONAL_ID_RESTART_SEEDS = (201, 202, 203, 204, 205)

const FUNCTIONAL_ID_REPORTING_CUTOFFS = (
    min_successful_restarts = 3,
    n_attempted_restarts = 5,
    traj_agree_rel_rmse = 0.05,
    d_disagree_scale_norm_rel_rmse = 0.20,
)

const FUNCTIONAL_ID_TRAINING_CONFIG = (
    adam = UNIQUE_CLAIM_PROTOCOL.adam_iterations,          # 100
    bfgs = UNIQUE_CLAIM_PROTOCOL.bfgs_iterations,          # 50
    frozen_phys = Symbol[],
    phys_init = nothing,
)
```

**L-SEEDS**

```
FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)
length(FUNCTIONAL_ID_RESTART_SEEDS) == 5
```

Tam olarak beş tohum: `(201, 202, 203, 204, 205)`.
Her tohum tam **bir** deneme. Yeniden deneme **yok**.
`103` fonksiyonel-ID restart’ı **değildir**.
M4 listesi `(103, 107, 111, 113, 127)` ile kesişim boştur.

`restart_seeds` argümanı:

- `length(restart_seeds) == 5` değilse → `ArgumentError`
  (sessiz `:incomplete` değil)
- `{201,202,203,204,205}` kümesi değilse → `ArgumentError`
- tekrar varsa → `ArgumentError`
- `103 ∈ restart_seeds` → `ArgumentError`

Beşten az tohum kabul eden her implementasyon reddedilir.
Üç restart’a gizli kısaltma yoktur.
Tohum 103’ü `201..205` etiketi altında kullanmak yoktur.

```
n_attempted  == 5
n_successful == count(included)
n_failed     == n_attempted - n_successful
complete     === (n_attempted == 5 && n_successful >= 3)
```

**L-CUT — raporlama kesitleri, kapı değil**

```
FUNCTIONAL_ID_REPORTING_CUTOFFS === (
    min_successful_restarts = 3,
    n_attempted_restarts = 5,
    traj_agree_rel_rmse = 0.05,
    d_disagree_scale_norm_rel_rmse = 0.20,
)
```

Bu bir **raporlama kuralı**dır, kapı değildir.
`FUNCTIONAL_ID_REPORTING_CUTOFFS` `RECOVERY_THRESHOLDS` **değildir**.
`FunctionalIdentifiability.jl` `RECOVERY_THRESHOLDS` okumaz.
`discovered_rate_rmse == 0.20` tuzak eşleşmesi gerekçe değildir.

Q4 `success` / `passed` kapısı **yoktur**.

---

## 9. Kilitli tipler ve türetilmiş bayraklar (L-FIELDS, L-DERIVE)

Dört struct. Alan listesi **kapalıdır**. Ek alan yok.

**Yasak alan adları (dört struct’ın hiçbirinde):**
`success`, `passed`, `identifiable`, `unidentifiable_edge`,
`collinearity`, `coefficients_are_biological_constants`,
`credible`, `holdout`, `payload`, `misc`, `extra`, `metadata`,
`data`, `any`. `Any` catch-all alan yok. Q4 success/passed kapısı
yoktur. Belirsizlik / hipotez / M4 occupancy alanı yoktur.

```julia
struct FunctionalIdentifiabilityDomain
    regulator_index::Int
    z::Vector{Float64}
    n_train_points::Int
    n_holdout_points::Int
    fill_value::Float64
    construction::Symbol   # yalnız :train_obs_union_holdout_obs
end

struct FunctionalIdentifiabilityRestart
    seed::Int
    included::Bool
    training_retcode::Union{TrainingRetcode,Nothing}
    failure_reason::Symbol
    message::String
    nn_init_fingerprint::UInt64
    nn_final_fingerprint::UInt64
end

struct FunctionalIdentifiabilityPair
    seed_i::Int
    seed_j::Int
    d_rmse_raw::Float64
    d_rmse_scale_normalized::Float64
    d_correlation::Float64
    scale_alpha::Float64
    traj_rmse_train::Float64
    traj_rmse_holdout::Float64
end
```

`FunctionalIdentifiabilityDiagnostic` alanları:

```julia
struct FunctionalIdentifiabilityDiagnostic
    family::Symbol
    restart_seeds::NTuple{5,Int}
    n_attempted::Int
    n_successful::Int
    n_failed::Int
    complete::Bool
    domain::FunctionalIdentifiabilityDomain
    restarts::Vector{FunctionalIdentifiabilityRestart}
    pairs::Vector{FunctionalIdentifiabilityPair}
    median_d_rmse_raw::Float64
    median_d_rmse_scale_normalized::Float64
    median_d_correlation::Float64
    median_traj_rmse_train::Float64
    median_traj_rmse_holdout::Float64
    trajectory_agree::Bool
    function_disagree::Bool
    trajectory_agree_function_disagree::Bool
    status::Symbol
    practical_not_structural::Bool   # her zaman true
end
```

`training_retcode::Union{TrainingRetcode,Nothing}` — fit öncesi
hata ⇒ `nothing`.

`failure_reason` yalnız:

```
:none
:fit_threw
:nonfinite_D
:nonfinite_trajectory
:predict_threw
```

`included == false` ⇒ `failure_reason != :none` ve `message != ""`
(L-FAILMSG). Başarısız mesajlar tutulur.

`status` yalnız (L-STAT):

```
:incomplete
:traj_disagree
:scale_ambiguity
:function_agree
:trajectory_agree_function_disagree
```

**Yasak status adları ve semantiği:**
`:structurally_identifiable`, `:functionally_identifiable`,
`:certified`, `:verified`, `:identifiable`, `:unidentifiable`.

`construction` alanı **kanıt değildir**. Canlı `z` ve canlı
`r_range` bağımsız doğrulanır (T-B-ZLIVE).

### L-DERIVE — bayraklar ve status bağımsız yazılamaz

`assess` Boolean bayrakları veya `status`’u serbestçe yazamaz.
`assess` imzası bu alanları keyword olarak **almaz**.

Tercih edilen tasarım:

`FunctionalIdentifiabilityDiagnostic` **bağımsız verilmiş**
`function_disagree` / `trajectory_agree` /
`trajectory_agree_function_disagree` / `status` değerlerini
kabul etmez.

Tek yasal montaj yolu
`assemble_functional_identifiability_diagnostic` (veya eşdeğer
iç kurucu) şunları **türetir**. T-C-DERIVE-LIVE bunu kurucu
üzerinden değil, canlı `assess` dönüşünün `pairs` alanından
yeniden türetir.

```
n_attempted  = 5
n_successful = count(r.included for r in restarts)
n_failed     = n_attempted - n_successful
complete     = (n_attempted == 5 && n_successful >= 3)

median_*     = Statistics.median(ilgili pair alanı)   # pairs boşsa NaN

trajectory_agree =
    complete &&
    median(traj_rmse_train)   ≤ 0.05 &&
    median(traj_rmse_holdout) ≤ 0.05

function_disagree === (
    complete &&
    n_successful >= 2 &&
    median(d_rmse_scale_normalized) >= 0.20
)

# incomplete / yorumlanamaz durumda function_disagree === false
# (dokümante incomplete/failed hâl yorumu engeller)

trajectory_agree_function_disagree =
    trajectory_agree && function_disagree

status =
    if !complete
        :incomplete
    elseif !trajectory_agree
        :traj_disagree
    elseif function_disagree
        :trajectory_agree_function_disagree
    elseif median_d_rmse_raw ≥ 0.20
        :scale_ambiguity
    else
        :function_agree
    end

practical_not_structural = true
```

`function_disagree` **yalnız** canlı `pairs` üzerinden

```
median(d_rmse_scale_normalized) ≥ 0.20
```

karşılaştırmasından gelir (complete / yorumlanabilir iken).
Yörünge metriği `function_disagree`’e **giremez**.

Kurucu / montaj `function_disagree` veya `status`’u serbest
parametre olarak alırsa, türetilmiş değerle çelişen her giriş
`ArgumentError` (veya eşdeğer red) fırlatır. Sessizce üzerine
yazmak “red” sayılmaz.

`assess` montajdan sonra `setfield!` ile bayrak değiştiremez.

`NaN ≥ 0.20` ve `NaN ≤ 0.05` false’tur.

Assemble / kurucu A/B/C testi T-C-DERIVE-LIVE’ın yerine
**geçmez**.

---

## 10. Sayısal NN parmak izi (L-FP)

Referans kimliği (`===`, `objectid`) **yetersizdir**.
`deepcopy` yeni referans üretir, aynı sayıları korur.
Kanıt sayısal parmak izidir.

```julia
nn_parameter_fingerprint(nn_params)::UInt64
```

- NN parametre yaprağını sabit sırada `Float64` vektöre indirger
- `hash` / kararlı tamsayı özeti döner
- `objectid` değildir
- tohum etiketi değildir
- pointer değildir
- `===` değildir

Aynı `MersenneTwister(seed)` + aynı `ude_net` ⇒ aynı init izi.
`seed=201` ve `seed=202` ⇒ **farklı** init izi (L-INIT).
`deepcopy(params)` ⇒ **aynı** final izi.

Başarısız restart, init öncesi ⇒ her iki iz `0x0`.
Init oldu, fit/predict düştü ⇒ init izi dolu, final `0x0`
veya fit dönmüşse final dolu.

---

## 11. Kilitli API

Unexported. `LOCKED_PUBLIC_EXPORTS` büyümez.

```julia
functional_identifiability_domain(
    split::ExperimentSplit, regulator::Integer;
    fill_value::Real = 0.3) -> FunctionalIdentifiabilityDomain

scale_align_destruction(D_i::AbstractVector, D_j::AbstractVector)
    -> (; alpha, D_j_aligned)

pairwise_destruction_metrics(D_i, D_j)
    -> (; d_rmse_raw, d_rmse_scale_normalized, d_correlation, scale_alpha)

pairwise_trajectory_metrics(
    pred_i_train, pred_j_train, pred_i_holdout, pred_j_holdout)
    -> (; traj_rmse_train, traj_rmse_holdout)

assemble_functional_identifiability_diagnostic(...)
    -> FunctionalIdentifiabilityDiagnostic
    # function_disagree / trajectory_agree / status kabul etmez; türetir

assess_functional_identifiability(
    split::ExperimentSplit,
    ude_net::BiologicalNetwork;
    restart_seeds = FUNCTIONAL_ID_RESTART_SEEDS,
    family::Symbol = :hill,
    adam = UNIQUE_CLAIM_PROTOCOL.adam_iterations,
    bfgs = UNIQUE_CLAIM_PROTOCOL.bfgs_iterations,
    frozen_phys = Symbol[],
    phys_init = nothing,
    fill_value = 0.3) -> FunctionalIdentifiabilityDiagnostic

format_functional_identifiability_diagnostic(
    diag::FunctionalIdentifiabilityDiagnostic) -> String

format_q3_q4_side_by_side(ident, diag) -> String
```

**L-TRUTH — `assess` imzasında yasak keyword / argüman:**
`truth_rate`, `D_true`, `truth_support`, `params`, `primary_fit`,
`trained`, `prefit`, `function_disagree`, `trajectory_agree`,
`trajectory_agree_function_disagree`, `status`. Canlı \(D\) yalnız

```
sample_unknown_destruction_grid(model, params, term; r_range = domain.z)
```

çıktısından gelir. Sentetik \(D\) vektörleri **yalnız** saf
metrik yardımcılarında (`scale_align_destruction`,
`pairwise_destruction_metrics`) verilir.

`fit_unknown_destruction` imzası **değişmez** (tercih edilen,
M2-güvenli yol). Opsiyonel `seed::Integer=0` eklenirse:

- varsayılan `0` M2 çağrılarını bit-özdeş bırakır
- bağımsızlık kanıtı sayılmaz
- casus hâlâ `build_ude_model(MersenneTwister(seed), ude_net)`
  ve bağımsız `p0` izini ister

İç `fit_functional_identifiability_restart` varsa yalnızca
`assess` çağırır; public/test owner değildir.

---

## 12. Canlı gözlem dikişleri (L-SPY)

PR testleri **canlı**

```
assess_functional_identifiability(...)
```

yolunu, çağrılan fonksiyonun **içinden** gözler. M2 casus
sözleşmeleri kırılmaz; M3 onlara ek dikiş ekler.

### 12.1 M2 sözleşmeleri (değişmez)

```
FIT_UNKNOWN_DESTRUCTION_OBSERVER
    observer(set) -> Union{TrainingResult, Nothing}

SAMPLE_UNKNOWN_DESTRUCTION_GRID_OBSERVER
    observer(r_range) -> Nothing
```

M2-G1 / L-FIT-A / L-DOM-* bu imzalara bağlıdır.

### 12.2 M3 fit-giriş gözlemi (L-FITSPY)

Mevcut fit casusu yalnız `ExperimentSet` görür. Bu, `p0`
bağımsızlığını veya params→sampler bağını kanıtlamaz.

`fit_unknown_destruction` **içinden**, eğitimden önce, M3
`FIT_UNKNOWN_DESTRUCTION_ENTRY_OBSERVER` (ad serbest; semantik
kilitli) her restart için kaydeder:

| Alan | Kaynak |
|---|---|
| `restart_seed` | çağrı sırası `k` ↔ `FUNCTIONAL_ID_RESTART_SEEDS[k]` (öz-bildirim değil; sıra kanıtı) |
| `p0` / `p0_nn` | gerçek `ude_p0` argümanı |
| `p0_fingerprint` | `nn_parameter_fingerprint(ude_p0.nn)` |
| `fit_set` | gerçek `set` |
| `fit_set_length` | `length(set)` |
| `fit_experiments_identity` | `set.experiments` nesne kimliği |
| `adam` | gerçek keyword |
| `bfgs` | gerçek keyword |
| `frozen_phys` | gerçek keyword |
| `phys_init` | gerçek keyword |
| `ordinal` | bu `assess` çağrısındaki fit-giriş sıra numarası |

Bu dikiş eğitimi **yutmaz** (`TrainingResult` döndürmez).
Yutma yalnız mevcut M2 `FIT_UNKNOWN_DESTRUCTION_OBSERVER`
üzerinden kalır (PR’da tam UDE yok).

Assess’in “ben seed=201 kullandım” yazması kanıt **değildir**.

Çağrı-sıra günlüğü (L-ORDER) aynı dikişte tutulur: her fit-giriş
olayı, herhangi bir holdout residual / `evaluate_holdout`
olayından **önce** damgalanır (T-B-HP-SENTINEL).

### 12.3 Bağımsız `p0` beklentisi (T-B-P0)

Test, her `k` için beklenen başlangıç NN parametrelerini
**kendisi** kurar:

```
build_ude_model(
    MersenneTwister(FUNCTIONAL_ID_RESTART_SEEDS[k]),
    ude_net
)[2].nn
```

ve sayısal parmak izini hesaplar.

Zorunlu:

```
live_p0_fingerprint[k] ==
    fingerprint(
        build_ude_model(
            MersenneTwister(FUNCTIONAL_ID_RESTART_SEEDS[k]),
            ude_net
        )[2].nn
    )
```

Tohum etiketi tek başına yeterli kanıt **değildir**.
T-B-P0 tek başına T-B-PARAMS’ın yerine geçmez.

### 12.4 M3 sample-sonuç + params gözlemi (L-SAMPSPY)

`sample_unknown_destruction_grid` **içinden**, dönüşten hemen
önce, M3 sonuç casusu kaydeder:

```
(; r_range, R, D, term, params, params_nn_fingerprint)
```

`D` 1×N Matrix olarak kaydedilir. Test `D = vec(D)` uygular.
`params` bu çağrıya gerçekten geçen `p` argümanıdır.
M2 `observer(r_range)` durur.

### 12.5 İkinci LIVE bağ: fit params → sampler → D (T-B-PARAMS)

`p0` testi, şu yanlışı öldürmez:

```
# YASAK IMPLEMENTASYON — T-B-PARAMS kırmızı
fit_once = fit_unknown_destruction(...)   # tek gerçek fit
for k in 1:5
    params_k = deepcopy(fit_once.params)  # veya kopya / aynı nesne
    label seed = 200 + k
    report restart k
end
```

Her restart `k` için test:

1. Restart `k`’nin gerçek `TrainingResult.params` değerini alır
   (casus gövdesinin döndürdüğü / fit’in döndürdüğü nesne).
2. Canlı `sample_unknown_destruction_grid` çağrısını içeriden
   gözler.
3. Sampler’ın kullandığı `params` parmak izini kaydeder.
4. Bağımsız olarak
   `sample_unknown_destruction_grid(model, fit_result[k].params,
   term; r_range = z_expected)` çağırır.
5. Bağımsız yeniden örneklenen \(D_k\) ile canlı örneklenen
   \(D_k\)’yi karşılaştırır.

Zorunlu:

```
live_sample_params_fingerprint[k] ==
    fingerprint(fit_result[k].params.nn)

live_D[k] == independently_sampled_D[k]
```

Casus fikstürü **beş sayısal olarak farklı** final parametre
seti verdiğinde ayrıca:

```
length(unique(live_D_fingerprints)) == n_successful
```

**Açıkça reddedilen:**

- bir final params’ın beş kez kopyalanması
- `deepcopy(params)` beş kez
- aynı params nesnesinin beş restart etiketi altında kullanılması
- farklı fit sonuçları, aynı sampled params
- farklı fit sonuçları, sampler’ın paylaşılan bir params
  nesnesi kullanması
- yalnız `===` nesne kimliği (parmak izi zorunlu)

### 12.6 M3 `predict_ude` çıktı gözlemi (L-PREDSPY, T-C-TBIND)

`predict_ude` (canlı `UDEModel` yöntemi) **içinden** M3-yerel
casus kaydeder:

```
(; params, u0, tspan, times, model, X)
```

`X` **gerçekten dönen** durum × zaman matrisidir.
Yalnız girdi `params` kaydı yetersizdir.
Yardımcının `predict_ude` çağırması yetmez.
Casus yokken `predict_ude` semantiği değişmez.

Fırlatmada `X` yoktur; T-B-PRED203 fırlatmayı kanıtlar.
Sessizce sonlu `X` uydurmak yasaktır.

T-C-TBIND, her dahil restart için:

```
predict_ude'nin döndürdüğü gerçek X
    → bağımsız yörünge metriği
    → canlı pair.traj_rmse_* 
```

sayısal bağını ister.

### 12.7 Assess giriş gözlemi

`assess_functional_identifiability` **içinden** giriş casusu
ateşlenir (T-G-CALL ve canlı zincir). Assess’i hiç çağırıp
tanı imal etmek bu casusu ateşlemez. Sayacı / gözlemci gerçek
yürütme kanıtıdır; string değildir.

---

## 13. Taze init (L-INIT)

Her restart, fit’ten **önce**:

```
model, p0 = build_ude_model(MersenneTwister(seed), ude_net)
```

`ude_net` (bilinmeyen yıkımlı ağ). `truth_net` değildir.
`consume_shared_suite_rng!` restart değildir.

L-FITSPY + bağımsız `build_ude_model` izi L-INIT’in kanıtıdır.
T-B-PARAMS olmadan tek-fit-kopya saldırısı yaşar.

---

## 14. Yalnız M2 train split (L-TRAIN)

Protokol / 7-IC testlerinde her restart yalnız

```
fit_unknown_destruction(model, p0, split.train; adam, bfgs,
                        frozen_phys, phys_init)
```

eğitir.

Zorunlu (7-IC fixture):

- `length(set) == 7`
- orijinal `Experiment` nesneleri
  (`set.experiments[k] === split.train.experiments[k]`)
- holdout nesnesi yok
- tam 9-IC fit yok
- train+holdout concatenation yok
- ikinci eğitmen yok
- restart başına **tam bir** `fit` çağrısı
- `attempt_count == 1`

Sentinel 4-nokta domain testi kendi train uzunluğunu kullanır
(T-B-ZLIVE); L-TRAIN ayrı 7-IC testidir.

---

## 15. Holdout yasağı (L-HP, T-B-HP-SENTINEL, T-B-INC-HOLD)

Holdout verisi **seçemez**:

- Adam iterasyonu
- BFGS iterasyonu
- optimizer konfigürasyonu
- `frozen_phys`
- `phys_init`
- restart dahil etme
- restart dışlama
- tohum seçimi
- restart seçimi
- en-iyi-restart seçimi
- domain seçimi
- çift filtreleme
- herhangi bir diğer eğitim kararı

Holdout değerlendirmesi (Q7 `evaluate_holdout` veya holdout
`hybrid_data_residual`) **yalnızca** beş restart eğitim
kararı — dahil etme elverişliliği dahil — belirlendikten
**sonra** olabilir. `assess` yolunda `evaluate_holdout`
zaten yoktur (T-E-WALK). Holdout residual / `evaluate_holdout`
beş fit kararından **önce** çağrılamaz.

Dahil etme, belgelenmiş M3 kriterleridir:

```
included == true  iff
    D sonlu
    train yörüngesi sonlu
    holdout yörüngesi sonlu
    failure_reason == :none
```

Holdout metriğinin büyüklüğü dahil etme **değildir**.

### T-B-INC-HOLD — holdout dahil etmeyi seçemez

Canlı sentinel restart:

- sonlu \(D\)
- sonlu yörünge (`predict_ude` sonlu `X` döner)
- **kasıtlı olarak kötü** holdout metriği
  (holdout residual / holdout yörünge hatası, herhangi
  makul eşiğin açıkça üstünde; örneğin `≥ 1e3`)

Beklenen:

```
included == true
```

çünkü dahil etme M3 kriterleridir, holdout kalitesi değil.

**Açıkça reddedilen:**

```
included = holdout_error < threshold
```

veya eşdeğer holdout-tabanlı filtre
(`evaluate_holdout`, holdout \(D\) RMSE, holdout traj RMSE
eşiği, “en iyi holdout restart”).

Test, holdout metriğinin kasten kötü olduğu bir vaka
kullanmak **zorundadır**. Aksi hâlde holdout-tabanlı
implementasyon tesadüfen yeşil kalır.

### T-B-HP-SENTINEL — holdout hiperparametre seçemez

Fit-giriş casusu eğitim konfigürasyonlarını **herhangi bir
holdout değerlendirmesinden önce** kaydeder.

Beş restart konfigürasyonu özdeş ve donmuş M3 protokolüdür:

```
frozen_m3_training_config = (
    adam = 100,
    bfgs = 50,
    frozen_phys = Symbol[],
    phys_init = nothing,
)

fit_config[k] == frozen_m3_training_config    # her k
```

Sentinel, **farklı** bir konfigürasyonu holdout-optimal yapar.

Örnek:

```
holdout sentinel: adam = 50  → holdout metriği daha iyi
canlı fit yine:   adam = 100
```

Zorunlu:

- hiçbir holdout residual veya `evaluate_holdout` çağrısı
  beş fit kararından önce yoktur
- gizli yardımcı ile seçim de reddedilir
- yalnız kaynakta `for adam in [...]` yokluğuna güvenilmez

**Açık kırmızı gövde:**

```julia
for adam in [50, 100, 500]
    fit(...)
    choose_using_holdout(...)
end

# ve
_select_training_by_holdout(split.holdout)  # gizli yardımcı
```

---

## 16. Ortak Q4 domain ve canlı doğrulama (L-DOMAIN, T-B-ZLIVE)

Domain, restart’a bağlı örneklemeden **önce**, **bir kez**
inşa edilir. Domain, restart sonuçlarından bağımsızdır.

Regülatör indeksi `build_ude_model` **olmadan** alınır:

```
compiled = compile_mechanism(ude_net)
regulator = tek NeuralDestructionTerm.regulator
```

Bu derleme restart değildir, `n_attempted`’e girmez, NN
parametresi üretmez / kullanmaz.

Sonra:

```
r_train   = concatenate(split.train  regulator observations
                        in original experiment and sample order)
r_holdout = concatenate(split.holdout regulator observations
                        in original order)
z         = vcat(r_train, r_holdout)
```

Zorunlu:

- önce train, sonra holdout
- tekrarlar korunur
- sıra korunur
- tam uzunluk korunur
- `sort` yok
- `unique` yok
- extrema indirgeme yok
- adaptif / post-hoc kırpma yok
- restart’a özel domain yok
- restart sonuçlarından türetilmiş domain yok
- `params` veya \(D\) anlaşmazlığından türetilmiş domain yok

`n_train_points == length(r_train)`
`n_holdout_points == length(r_holdout)`
`length(z) == n_train_points + n_holdout_points`

`fill_value = 0.3` (M2 grid sözleşmesi).

`construction === :train_obs_union_holdout_obs` her domain’de
yazılır ama **kanıt değildir**.

### T-B-ZLIVE — zorunlu sentinel ve bağımsız `z_expected`

Test, regülatör gözlemleri tam olarak

```
[0.1, 0.5, 0.1, 0.8]
```

olan bir sentinel split kurar (orijinal deney / örnek sırası).

Referans inşa (test tarafından, assess’ten bağımsız):

```
# Train:  deney T1 regulator [0.1], deney T2 regulator [0.5]
# Holdout: deney H1 regulator [0.1], deney H2 regulator [0.8]

regulator = testin bağımsız derlediği unknown-destruction regulator

r_train   = vcat(T1.observations[regulator, :],
                 T2.observations[regulator, :])
r_holdout = vcat(H1.observations[regulator, :],
                 H2.observations[regulator, :])
z_expected = vcat(r_train, r_holdout)
# == [0.1, 0.5, 0.1, 0.8]
```

Canlı `assess` yolu gözlenir. Test ister:

```
her gerçek sample_unknown_destruction_grid(...) çağrısının
    r_range == z_expected

ve

diagnostic.domain.z == z_expected
```

Bu değerler test tarafından bağımsız doğrulanır.
`construction` alanı yeterli değildir.

**Açıkça reddedilen:**

- `sort(z)`
- `unique(z)`
- varsayılan `range(0.05, 2.0)`
- `_regulator_grid(...)`
- `_unique_claim_external_regulator_band(...)`
- sabit keyfi 80-nokta ızgara
- restart’a özel domain
- restart sonuçlarından domain
- `params` veya \(D\) anlaşmazlığından domain
- post-hoc kırpma

Protokol verisinde (9 IC × 50 nokta) beklenen uzunluk
`350 + 100 = 450`’dir (ayrı protokol testi). Sentinel kendi
4 uzunluğunu korur.

**Yasak domain kaynakları (dosya + T-E-WALK):**
`_regulator_grid`,
`_unique_claim_external_regulator_band`,
`range(0.05, 2.0`,
`range(0.0, 1.0`.

---

## 17. Domain değişmezliği (L-ZIMM)

Domain şu çağrılardan **önce** kurulur:

- restart `build_ude_model`
- `fit_unknown_destruction`
- `sample_unknown_destruction_grid`
- `predict_ude`
- çift inşası

Restart sonuçlarını değiştirmek `z`’yi değiştirmez.
İki başarılı restart ile beş başarılı restart **aynı** `z`
kullanır (bit-özdeş `==`). Tüm çift metrikleri aynı
`domain.z` üzerindedir.

`functional_identifiability_domain` imzasında `D`, `params`,
restart listesi yoktur.

---

## 18. Canlı D örnekleme ve kaynak (L-DSAMP, T-C-DBIND, T-C-DSOURCE)

Canlı restart yolu **zorunlu** çağırır:

```
(R, D_matrix, term_out) = sample_unknown_destruction_grid(
    model, params, term;
    r_range = domain.z)
D = vec(D_matrix)
@assert length(D) == length(domain.z)
```

`term = only_unknown_destruction(model)`.

Canlı Q4 pairwise metrikleri **gerçek örneklenmiş nöral \(D\)**
ile bağlıdır.

L-SAMPSPY her başarılı restart için gerçek çıktıyı kaydeder.
Test `D = vec(D)` uygular ve beklenen pairwise metrikleri
**bağımsız** hesaplar:

```
alpha = dot(D_i, D_j) / dot(D_j, D_j)     # D_j == 0 ⇒ NaN
d_rmse_raw = rate_rel_rmse(D_i, D_j)
d_rmse_scale_normalized = rate_rel_rmse(D_i, alpha .* D_j)
```

Tanı çift metrikleri bu bağımsız yeniden hesapla
tam / uygun eşitlikte olmalıdır (T-C-DBIND).

**Birincil Q4 \(D\) kaynağı değildir (T-C-DSOURCE):**

- `equation_to_function`
- `normalize_destruction_samples`
- sembolik \(D\)
- normalize \(D\)
- truth \(D\) enjeksiyonu
- keyfi \(D\) değerleri
- sabit kodlanmış çift metrikleri
- gerçek \(D\) örneklemesinden **önce** hesaplanmış metrikler
- canlı sampler’dan bağımsız uydurulmuş pairwise sayılar
- varsayılan `0.05:2.0`
- `_regulator_grid`
- dış bant
- restart’a özel ızgara

`sample_destruction` ancak `r_range = domain.z` ile
çağrılırsa kabul; tercih edilen isim
`sample_unknown_destruction_grid`’dir.

Canlı `assess` truth \(D\) kabul etmez.

Çiftler arası interpolasyon yoktur. Her \(D[k]\)
`domain.z[k]` içindir.

T-C-DSOURCE, T-C-DBIND’in “sayılar uyuyor” iddiasını
**kaynak** olarak kilitler: canlı çift, canlı sample
çıktısından gelir; `assess` sonra başka bir \(D\) uyduramaz.

---

## 19. Canlı `predict_ude`, X bağ ve yalıtım (L-PRED, T-C-TBIND, T-B-PRED203)

Runtime test kanıtlar:

```
assess_functional_identifiability
    → build_ude_model
    → fit_unknown_destruction
    → sample_unknown_destruction_grid
    → predict_ude
```

bunların **canlı yolda** gerçekleştiğini. Yardımcının
`predict_ude` çağırması yetmez.

```
X = predict_ude(params, e.u0,
                (first(e.times), last(e.times)),
                e.times, model)
```

Train ve holdout deneylerinin her biri için (dahil edilen
restart’ta). Dönen şekil durum × zaman. Casus bu `X`’i
kaydeder.

Yörünge metriği (Q1 `hybrid_data_residual` **değil**):

```
ρ_e = rate_rel_rmse(vec(X_i), vec(X_j))
traj_rmse_train_ij   = mean(ρ_e for e in split.train)
traj_rmse_holdout_ij = mean(ρ_e for e in split.holdout)
```

T-C-TBIND: her dahil restart’ın canlı dönen `X`’inden
bağımsız yeniden hesap, `pair.traj_rmse_*` ile eşleşir.

**Reddedilen:**

- uydurulmuş yörünge metrikleri
- truth’tan hesaplanmış yörünge metrikleri
- başka restart’tan kopyalanmış metrikler
- sabit sabitten atanmış metrikler
- yalnız girdi-params casusu (dönen `X` yok)

### T-B-PRED203 — zorunlu yalıtım testi

Tohum **203** için ayrılmış prediction-failure fikstürü.

Kanıtlar:

- `predict_ude` **gerçekten fırlatır** (L-PREDSPY içinden;
  203’ün gerçek `params` parmak izine bağlı)
- `assess` **fırlatmaz** / abort etmez
- tohum 203 `restarts` içinde görünür
- `included == false`
- `failure_reason == :predict_threw`
- `message != ""`
- `n_attempted == 5`
- yeniden deneme yoktur (`attempt_count[203] == 1`)
- kalan restart’lar devam eder

**Yasak:**

- bir prediction fırlatması yüzünden tüm Q4 değerlendirmesini
  abort etmek
- fırlatmayı sessizce kabul edilmiş restart’a çevirmek
- fırlatmayı sessizce sonlu çıktıya çevirmek
- başarısız restart’ı silmek
- tohum 203’ü yeniden denemek
- başarısızlığı sahte sonlu yörünge ile değiştirmek

---

## 20. NotConverged politikası (L-RETCODE)

`TrainingRetcode` başarı filtresi **değildir**.

`TrainingRetcode.NotConverged` + sonlu yörüngeler + sonlu \(D\)
bir restart’ı **otomatik dışlamaz**.

```
TrainingRetcode == NotConverged
D sonlu
train yörüngesi sonlu
holdout yörüngesi sonlu
⇒ included == true
```

Retcode kaydedilir. Yalnız `Success` kabul etmek geçerli
negatifleri gizler; test bunu reddeder.

T-B-INC-HOLD ile karışmaz: NotConverged sonluluk meselesidir;
holdout kalitesi değildir.

---

## 21. Tam muhasebe (L-ACCT)

```
n_attempted  == 5
n_successful == count(included)
n_failed     == n_attempted - n_successful
complete     === (n_attempted == 5 && n_successful >= 3)
length(restarts) == 5
length(restart_seeds) == 5          # aksi ArgumentError
```

Tohum başına tam bir deneme. Yeniden deneme yok.
`restarts[k].seed == restart_seeds[k]` (verilen sıra).

Üç başarılı + iki denememiş ⇒ `complete=false` (zaten
`n_attempted` 5 değilse `ArgumentError`).

---

## 22. Pairwise tamlık (L-PAIR)

`n = n_successful` için

```
length(pairs) == binomial(n_successful, 2) == n*(n-1)÷2
```

Her sırasız çift tam bir kez. `seed_i < seed_j`.

Tüm çift metrikleri **aynı** `domain.z` kullanır.

**Reddedilen:**

- self çift `(s,s)`
- ters mükerrer `(j,i)`
- yüksek hatalı başarılı çiftin düşülmesi
- yalnız-medyan özet (çiftler silinerek)
- çifte özel domain
- çifte özel interpolasyon

`n_successful < 2` ⇒ `pairs` boş; medyanlar `NaN`;
`complete=false`; bayraklar false.

---

## 23. Tam LS hizası (L-LS)

LS ölçeklemesi **zorunlu**:

```
alpha = dot(D_i, D_j) / dot(D_j, D_j)
```

payda `> 0` iken. `D_j ==` sıfır vektör ⇒ `alpha = NaN` ve
`d_rmse_scale_normalized = NaN`. Çift silinmez.
Keyfi `eps` paydası ile sıfır vektörü sonlu yapmak yasaktır.

Ayrımcı test (yardımcı + canlı):

```
D1 = [1, 2, 3]
D2 = [2, 4, 9]

alpha = 37/101

D2_aligned = alpha .* D2
d_rmse_scale_normalized = rate_rel_rmse(D1, D2_aligned)
```

**Açıkça karşılaştırılıp reddedilen yanlışlar:**

- ters LS: `dot(D1, D2) / dot(D1, D1)` ⇒ `37/14`
- max-abs normalizasyon: `α = 3/9`

```
D2 = [0, 0, 0]
⇒ alpha = NaN
⇒ d_rmse_scale_normalized = NaN
```

Yardımcı `D_j=0` testi T-C-ZERO-LIVE’ın yerine **geçmez**.

`d_rmse_raw = rate_rel_rmse(D1, D2)` ölçek-duyarlı refakatçidir;
`function_disagree` onu okumaz.

`d_correlation = cor(D1, D2)`; `NaN → 0.0` saklanan `Float64`
için. Şekil kararı değildir.

`scale_align_destruction(D_i, D_j)` yönü: \(j \to i\),
payda \(\langle D_j, D_j\rangle\).

---

## 24. Fonksiyon anlaşmazlığı ⊥ yörünge (L-INDEP, T-C-DERIVE-LIVE)

Birincil fonksiyon anlaşmazlığı ölçütü:

```
median(d_rmse_scale_normalized) >= 0.20
```

Yörünge kesiti:

```
traj_agree_rel_rmse = 0.05
```

`RECOVERY_THRESHOLDS` kullanılmaz.

Üç hücre **canlı `assess`** yolundan geçer (casus kontrollü
\(D\) ve \(X\); sahte restart + yalnız `assemble` yetmez).

| Hücre | Yörünge | Ölçek-norm \(D\) | `trajectory_agree` | `function_disagree` | `trajectory_agree_function_disagree` |
|---|---|---|---|---|---|
| A | yakın (≤0.05) | yakın (<0.20) | true | false | false |
| B | yakın (≤0.05) | uzak (≥0.20) | true | true | true |
| C | uzak (>0.05) | uzak (≥0.20) | false | true | false |

```
A: trajectory near, D near
   ⇒ function_disagree = false

B: trajectory near, D far
   ⇒ function_disagree = true
   ⇒ trajectory_agree_function_disagree = true

C: trajectory far, D far
   ⇒ trajectory_agree = false
   ⇒ function_disagree = true
```

T-C-DERIVE-LIVE, canlı `assess` dönüşünden:

```
complete
n_successful
median(d_rmse_scale_normalized)     # live pairs
median trajectory metric            # live pairs
```

türetir ve ister:

```
function_disagree === (
    complete &&
    n_successful >= 2 &&
    median(d_rmse_scale_normalized) >= 0.20
)
```

(dokümante incomplete/failure hâli yorumu engellemiyorsa).

`trajectory_agree` belgelenmiş yörünge kesiti ve canlı çift
metriklerinden türetilir.

`status` şunlardan türetilir:

- complete
- yörünge anlaşması
- fonksiyon anlaşması / anlaşmazlığı
- belgelenmiş M3 durum kuralları

**Reddedilen:**

```
function_disagree = false          # canlı pair median ≥ 0.20 iken
function_disagree = trajectory_agree
status = :structurally_identifiable
```

`assess` bu alanlar için bağımsız Boolean kabul etmez.

---

## 25. Truth-D yasağı (L-TRUTH)

Canlı `assess` API `truth_rate` / `D_true` / `truth_support`
almaz. Canlı assess truth \(D\) kabul etmez.

T-C-DBIND + T-C-DSOURCE, \(D\)’nin
`sample_unknown_destruction_grid(model, params, term)` çıktısı
olduğunu kanıtlar.

---

## 26. M4 / keşif yasağı ve gizli yürüyüş (L-NOM4, T-E-WALK)

`FunctionalIdentifiability.jl` **ve**
`assess_functional_identifiability`’den erişilebilir tüm
yardımcılar, M2-G1 ile **aynı geçişli yerel çağrı-grafı**
standardını kullanır (L-DISC-B-2 gücü).

**Giriş noktası (ENTRY):** `assess_functional_identifiability`

Sözleşme (dar AST / çağrı-grafı; jenerik kaynak-string
envanteri yeterli değildir):

1. `assess_functional_identifiability` girişini tanır
2. Ondan erişilebilir yerel yardımcı çağrılarını statik çözer
3. Bu yardımcı gövdelerini **geçişli** inceler
4. Yasaklı keşif / holdout-seçim / yasak domain yoluna
   giden herhangi bir kenar varsa kırmızı
5. Bir-düzey veya çok-düzey yardımcı dolayımını reddeder

Gizli yardımcı kaçış **değildir**.

Yasak hedefler (doğrudan veya geçişli yol):

```
discover_unknown_rate
discover_equations
discover_unknown_destruction
run_recovery_suite
_train_unknown_edge
_evaluate_unknown_rate_recovery
evaluate_holdout
report_recovery
_regulator_grid
_unique_claim_external_regulator_band
range(0.05, 2.0
range(0.0, 1.0
```

Ayrıca `FunctionalIdentifiability.jl` dosya gövdesinde bu
token’lar yoktur. Dosya-token taraması T-E-WALK’ın yerine
geçmez: `assess` gövdesi temiz görünüp işi bir yardımcıya
devretmek kaçış **değildir**.

Onaylı yapraklar (yürüyüş bunlara **girmez**):
`build_ude_model`, `fit_unknown_destruction`,
`sample_unknown_destruction_grid`, `predict_ude`,
`only_unknown_destruction`, `compile_mechanism` ve saf
metrik yardımcıları. Bu yaprakların kendi iç grafı Q4
yasağı değildir.

`Project.toml` `StructuralIdentifiability`, `Turing`, `MCMC`
içermez.

Öldürür (T-E-WALK kırmızı; dosya token’ı temiz olsa bile):

```
assess_functional_identifiability(...)
    → _q4_helper(...)
        → discover_unknown_rate(...)

assess_functional_identifiability(...)
    → _prepare(...)
        → _forward(...)
            → discover_unknown_destruction(...)

assess_functional_identifiability(...)
    → helper(...)
        → evaluate_holdout(...)

assess_functional_identifiability(...)
    → _domain_helper(...)
        → _regulator_grid(...)
```

---

## 27. PR maliyet sınırı (L-COST)

Beş-restart gerçek UDE eğitimi **eklenmez:**

- `test/runtests.jl` (pahalı çağrı; ucuz include serbest)
- M2 hard job (`test/run_recovery_hard.jl`)
- PR `recovery` job
- PR `test` job

PR testleri casus + sahte `TrainingResult` kullanır.
Gerçek beş-restart UDE benchmark / nightly’dedir; PR’a eklenmez.

```
!occursin("assess_functional_identifiability", "test/run_recovery_hard.jl")
!occursin("functional_identifiability.jl", ".github/workflows/ci.yml")
!occursin("assess_functional_identifiability", "test/runtests.jl")
```

`runtests.jl` `include("test_functional_identifiability.jl")`
içerebilir; o dosya tam UDE çalıştırmaz.

Format listesine yeni dosya eklemek serbesttir (hijyen).
Pahalı iş eklemek yasaktır.

---

## 28. Formatter (L-FMT, L-FAILMSG, L-NEG)

`format_functional_identifiability_diagnostic` basmak zorundadır:

- her restart
- her sırasız çift
- her çift \(D\) metriği
- her çift yörünge metriği
- başarısızlık tohumu
- `failure_reason`
- `message`
- medyanlar
- `status`
- `complete`

Yalnız-medyan özet yasaktır.
Yüksek hatalı başarılı çift filtresi yasaktır.

`included == false` satırı: tohum + neden + mesaj.

Beş status için format fixture zorunlu. Her biri şunları içerir:

- `practical functional diagnostic`
- `not a structural identifiability certificate`
- `not a unique-claim gate`

**Yasak kullanıcı cümleleri:**
`functionally identifiable`, `structurally identifiable`,
`structural identifiability certificate`,
`Bayesian credible`, `Q4 gate`, `certified`, `verified`.

`format_protocol_result` / `PROTOCOL_PRINT_FIELDS` değişmez.
Unique-claim stdout Q4 bloğu almaz.
`report_recovery` / `format_protocol_result` kaynağında
`assess_functional`, `format_functional`, `format_q3_q4` yoktur.

---

## 29. Fisher terminolojisi (L-FISH)

Kullanıcı yüzünde:

| Yasak | Zorunlu |
|---|---|
| credible interval | asymptotic Fisher interval |
| credible level | nominal coverage |
| Bayesian / Bayes | (yok; yöntem Bayes değildir) |

`parameter_credible_intervals` **unexported** adı kırılmak
zorunda değildir. Kullanıcıya dönük metin M3’te retarget edilir:

- `IdentifiabilityReport` docstring (“credible intervals are local”)
- `parameter_credible_intervals` docstring
  (“Asymptotic normal credible intervals…”)
- `_z_score` hata metni (“unsupported credible level”)

Honesty / kaynak testi zorunludur:
`src/Identifiability.jl` kullanıcı cümlesi “Bayesian” /
“credible interval” / “credible level” demez; “asymptotic
Fisher interval” ve “nominal coverage” der.

Yeni interval alanı yoktur. Q4 struct’ında interval yoktur.

---

## 30. Gerçek benchmark yolu (L-BENCH, T-G-CALL)

`benchmark/functional_identifiability.jl` **gerçekten**

```
assess_functional_identifiability(...)
```

**çağırır** ve dönen tanıyı kullanır (formatlar).

M3 benchmark gerçektir; uydurma tanı tablosu değildir.
M3 complete **gerçek script’i içerir**. Nightly çizelgeleme
M7’de kalabilir. Script yokken veya `assess`’i çalıştırmadan
M3 complete sayılmaz.

String `contains` yeterli **değildir**.
Yorum, kullanılmayan referans, doğrudan
`FunctionalIdentifiabilityDiagnostic(...)` kurulumu ve sahte
benchmark çıktısı kabul **edilmez**.

T-G-CALL gerçek gözlemci / çağrı sayacı (veya eşdeğer yürütme
kanıtı) kullanır. Script, M3 casusları (sahte `TrainingResult`,
ucuz predict) altında **çalıştırılır**. Assess giriş casusu
ateşlenmeden test kırmızı. Beş gerçek UDE PR’a eklenmez.

Nightly generate (script; PR değil):

```
generate_recovery_experiments(
    MersenneTwister(FUNCTIONAL_ID_DATA_SEED),
    truth_net, truth_params;
    tspan = UNIQUE_CLAIM_PROTOCOL.tspan,
    n_points = UNIQUE_CLAIM_PROTOCOL.n_points,
    noise_σ = UNIQUE_CLAIM_PROTOCOL.observation_noise)
unique_claim_experiment_split(set)
assess_functional_identifiability(split, ude_net)
format_functional_identifiability_diagnostic(diag)
```

`consume_shared_suite_rng!` yok. Suite RNG’sine bit-özdeşlik
zorunlu değildir.

---

## 31. Sıfır payda canlı testi (T-C-ZERO-LIVE)

Yardımcı `D_j = 0` testi yeterli **değildir**.

Canlı testte bir canlı restart çifti:

```
D_j = [0, 0, ..., 0]
```

Zorunlu:

```
alpha = NaN
d_rmse_scale_normalized = NaN
```

`assess` çıktısı çifti **korur**. Yapamaz:

- `eps` enjekte etmek
- `alpha = 1` yerine koymak
- çifti sessizce düşmek
- çifti sonlu metrik ile başarılı işaretlemek

---

## 32. Canlı yol koruması (L-LIVE)

Runtime test kanıtlar:

```
assess_functional_identifiability
    → build_ude_model
    → fit_unknown_destruction
    → sample_unknown_destruction_grid
    → predict_ude
```

Yardımcı testleri tek başına yetersizdir. `assess` stub +
doğru yardımcılar “complete” sayılmaz. §39 kataloğu bu
kuralın somut testleridir.

---

## 33. Saldırı → kilit / test haritası

Her satır yazılı teste bağlıdır. Öz-bildirim yeterli değildir.

| # | Saldırı | Öldüren |
|---|---|---|
| 1 | Tek UDE fit, final params ×5 kopya/`deepcopy`, etiket 201..205 | T-B-PARAMS + `unique(live_D_fingerprints)` |
| 2 | Aynı params, farklı etiket | T-B-PARAMS parmak izi |
| 3 | Farklı fit, aynı sampled params / paylaşılan sampler params | T-B-PARAMS `live_sample_params_fingerprint` |
| 4 | Aynı NN init / yalnız seed etiketi | T-B-P0 `live_p0_fingerprint[k] == build_ude_model(...)` |
| 5 | Seed 103 init’i `201..205` altında | T-B-P0 |
| 6 | Restart holdout görür | L-TRAIN casus length==7 |
| 7 | Holdout ile HP seçimi / gizli yardımcı | T-B-HP-SENTINEL |
| 8 | Holdout ile dahil etme | T-B-INC-HOLD |
| 9 | Restart’a özel domain | T-B-ZLIVE `r_range == z_expected` |
| 10 | Domain \(D\) sonrası | L-ZIMM sıra |
| 11 | Domain başarıya göre değişir | 2 vs 5 başarı aynı `z` |
| 12 | `sort(z)` / `unique(z)` | sentinel `[0.1,0.5,0.1,0.8]` |
| 13 | Varsayılan `0.05:2.0` / 80 nokta | T-B-ZLIVE |
| 14 | `_regulator_grid` / dış bant | T-B-ZLIVE + T-E-WALK |
| 15 | `construction` yalanı | T-B-ZLIVE bağımsız `z` |
| 16 | max-abs birincil şekil | L-LS `37/101` ≠ `3/9` |
| 17 | Ters LS | L-LS `37/101` ≠ `37/14` |
| 18 | Sıfır payda sessiz (yardımcı doğru, canlı yanlış) | T-C-ZERO-LIVE |
| 19 | Yörünge = fonksiyon; canlı bayrak uydurma | T-C-DERIVE-LIVE A/B/C |
| 20 | `function_disagree=false` iken median ≥0.20 | T-C-DERIVE-LIVE |
| 21 | `function_disagree = trajectory_agree` | T-C-DERIVE-LIVE hücre B |
| 22 | `status = :structurally_identifiable` | T-C-DERIVE-LIVE |
| 23 | Sembolik / normalize / truth \(D\) | T-C-DBIND + T-C-DSOURCE |
| 24 | Çift metrikleri sampler’dan önce | T-C-DBIND |
| 25 | Uydurma / truth / kopya yörünge metriği | T-C-TBIND canlı `X` |
| 26 | Kötü başarılı çift atılır | `binomial(n,2)` |
| 27 | Başarısız `n_attempted` dışı | `n_attempted==5` |
| 28 | Başarıya kadar retry | `attempt_count==1` |
| 29 | 203 fırlatması tüm Q4’ü düşürür / yutar | T-B-PRED203 |
| 30 | NotConverged filtre | L-RETCODE |
| 31 | `n_successful` şişer | `count(included)` |
| 32 | 2 başarı = complete | `complete` formülü |
| 33 | Q4 gizli kapı | hold/hard okumaz |
| 34 | `RECOVERY_THRESHOLDS` değişir | değer kilidi |
| 35 | Q4 `MechanismRecoveryResult` | T-E-M2HASH |
| 36 | Q4 `report_recovery` / `evaluate_holdout` / suite | T-E-WALK |
| 37 | Public export | `∉ names` + LOCKED |
| 38 | StructuralIdentifiability.jl / Bayes aralık | L-NOM4 + L-FISH |
| 39 | “credible” Bayes okunur | L-FISH honesty |
| 40 | M4 occupancy / `discover_*` Q4 (gizli yardımcı) | T-E-WALK |
| 41 | Dummy time Q4 girdi | `range(0.0,1.0` yok |
| 42 | Benchmark yorum / sahte tanı | T-G-CALL sayacı |
| 43 | 5 UDE her PR | L-COST |
| 44 | Negatif gizlenir | L-NEG + 5 status format |
| 45 | Yalnız medyan | L-FMT her çift |
| 46 | Teorem status | L-STAT küme |
| 47 | Bayrak `complete` olmadan | L-DERIVE |
| 48 | `m≠5` sessiz | `ArgumentError` |
| 49 | M2 test zayıflar | T-E-M2HASH; `∉ names` |
| 50 | `success`/`passed` / `Any` kova | L-FIELDS |
| 51 | Unique-claim stdout | format_protocol durur |

---

## 34. Dilimler M3-A … M3-H

M3-A…H **iç uygulama dilimleridir**. Public API dilimi açılmaz.

Her dilimde: dosyalar, sorumluluklar, dışlamalar, testler
(beklenen değer + kırmızı gövde), rollback, kabul, bağımlılık.

LIVE-path testlerin tam sözleşmesi §39’dadır. Dilim tabloları
o sözleşmeye bağlanır; onu gevşetmez.

---

### M3-A — Saf domain + metrikler

**Dosyalar**

- yeni `src/FunctionalIdentifiability.jl` (tipler, sabitler,
  `functional_identifiability_domain`, `scale_align_destruction`,
  `pairwise_*`, `nn_parameter_fingerprint`,
  `assemble_functional_identifiability_diagnostic`)
- `src/BioDynaX.jl` — `include` (`RecoveryPipeline.jl` sonrası);
  **export yok**
- `test/test_holdout.jl`, `test/test_recovery_pipeline.jl` —
  yalnız
  `!isdefined(..., :FunctionalIdentifiabilityDiagnostic)` →
  `FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)`
  (include-engeli; başka M2 satırı değişmez)
- yeni `test/test_functional_identifiability.jl` (A testleri)
- `test/runtests.jl` — ucuz include
- `test/internals.jl` — gerekirse unexported isimler

**Sorumluluklar**

- L-FIELDS, L-CUT, L-SEEDS, L-DERIVE (sentetik montaj)
- L-DOMAIN saf `domain()` (eğitim yok)
- L-LS, L-INDEP sentetik vektör / sahte çift montajı
- `nn_parameter_fingerprint`
- L-M2INT retarget (`∉ names`)

**Dışında**

- `build_ude_model` restart döngüsü
- `fit_unknown_destruction`
- canlı `assess` (C’de owner kapanır; A iskelet kurucu serbest
  değildir — bayrak yazan kurucu yok)
- keşif, holdout, suite
- gerçek UDE
- M2-G1/G2 veya L-* M2 testlerini `isdefined` ile değiştirmek
- T-C-DERIVE-LIVE / T-C-ZERO-LIVE / T-C-TBIND’i A’da “bitti”
  saymak

**Testler**

| ID | Canlı / kaynak | Bağımsız beklenti | Kırmızı gövde |
|---|---|---|---|
| T-A-SENTINEL | `functional_identifiability_domain(sentinel_split, regulator)` | `z == [0.1, 0.5, 0.1, 0.8]` | `sort(z)`, `unique(z)`, extrema |
| T-A-ORDER | aynı | train sonra holdout; tekrarlar durur | holdout-first; `unique` |
| T-A-LS | `scale_align_destruction([1,2,3],[2,4,9])` | `alpha == 37/101` | `dot(D1,D2)/dot(D1,D1)` (`37/14`); max-abs `3/9` |
| T-A-ZERO | `D2=[0,0,0]` | `alpha` ve scale-norm `NaN` | `eps` / `α=1` |
| T-A-SCALE | \(D_2=2D_1\) | scale-norm ≈ 0, raw yüksek | ham RMSE = `function_disagree` |
| T-A-ABC | `assemble(...)` hücre A/B/C | §24 tablosu | `function_disagree=trajectory_agree` |
| T-A-DERIVE | montaj `function_disagree`/`status` almaz | türetilmiş bayraklar | serbest bayrak yazımı |
| T-A-FLAG-REJECT | tutarsız elle bayrak | `ArgumentError` | `median≥0.20` iken `function_disagree=false` kabul |
| T-A-FIELDS | `fieldnames` tam eşitlik | kilitli tuple | `success` / `payload` / `Any` ekleme |
| T-A-STATUS | izinli status kümesi | beş sembol | `:certified` / `:functionally_identifiable` |
| T-A-CUT | cutoff `===` | tam named tuple | `0.20 → 0.80` |
| T-A-SEEDS | `(201,202,203,204,205)` | `===` | `103` ekleme; uzunluk ≠5 |
| T-A-NAMES | tip `∉ names(BioDynaX)` | unexported | export |
| T-A-M2NAMES | M2 dosyaları `∉ names` kullanır | `!isdefined` yok | zayıf `isdefined` bırakmak |

T-A-ZERO ve T-A-ABC yardımcı/kurucu testleridir. Canlı
karşılıkları T-C-ZERO-LIVE ve T-C-DERIVE-LIVE olmadan
kabul **edilmez**.

**Rollback:** include’u kaldır; `∉ names` retarget’i geri al
(yalnız bu dilim geri alınırsa ve tip silinirse).

**Kabul:** T-A-* yeşil; eğitim yok; export yok; bayraklar
türetilir; M2 `∉ names` retarget’i include ile aynı anda.

**Bağımlılık:** M2 complete.

---

### M3-B — Taze restart eğitimi

**Dosyalar**

- `src/FunctionalIdentifiability.jl` — per-restart gövde:
  init, bir `fit(split.train)`, grid, `predict_ude`, yalıtım
- `src/RecoveryPipeline.jl` — **yalnız** M3 fit-giriş dikişi;
  M2 `observer(set)` imzası durur
- `src/Recovery.jl` — **yalnız** sample sonuç + params dikişi;
  M2 `observer(r_range)` durur
- `src/Training.jl` — **yalnız** `predict_ude` içi M3 casusu
  (dönen `X` dahil); casus yokken davranış değişmez
- `test/test_functional_identifiability.jl` — canlı `assess`
  testleri

**Sorumluluklar**

- L-INIT, L-FITSPY, L-TRAIN, L-HP, L-DSAMP, L-ZLIVE,
  L-PRED, L-PRED203, L-RETCODE, L-PREDSPY, L-SAMPSPY
- T-B-P0, T-B-PARAMS, T-B-HP-SENTINEL, T-B-INC-HOLD,
  T-B-PRED203, T-B-ZLIVE
- `try/catch` per restart
- `D = vec(D_matrix)`, `length(D)==length(z)`
- `r_range = domain.z`
- `NotConverged` + sonlu ⇒ `included=true`
- tohum başına bir deneme; retry yok
- holdout değerlendirmesi beş eğitim + dahil etme
  kararından sonra (assess yolunda `evaluate_holdout` yok)

**Dışında**

- çift / status montajı (C) — B yalıtılmış restart kayıtları
  üretebilir; serbest `function_disagree` yazamaz
- `evaluate_holdout` / keşif
- `fit_unknown_destruction` imza değişikliği (tercih)
- 5× protokol UDE (PR)
- M2 casus imzalarını kırmak

**Testler**

| ID | Canlı gözlem | Bağımsız beklenti | Kırmızı gövde |
|---|---|---|---|
| T-B-LIVE | assess içinden build→fit→sample→predict | her adım casus sayacı >0 | stub `assess`; yardımcı-only predict |
| T-B-P0 | §39 | `live_p0_fingerprint[k] == fingerprint(build_ude_model(...))` | tek `p0`; 103 reuse; yalnız etiket |
| T-B-PARAMS | §39 | `live_sample_params_fingerprint[k] == fingerprint(fit_result[k].params.nn)` ve `live_D[k] == independently_sampled_D[k]`; `length(unique(live_D_fingerprints)) == n_successful` | tek fit ×5 kopya/`deepcopy`; paylaşılan sampled params |
| T-B-INIT | 201≠202 init izi | iki bağımsız `build_ude_model` izi | `MersenneTwister(0)` her seferinde |
| T-B-TRAIN | fit-giriş `set` kimliği | `length==7`, `set.experiments[k]===split.train.experiments[k]` | 9-IC / vcat / holdout nesnesi |
| T-B-ONEFIT | tohum başına 1 fit | `attempt_count==1` | retry / ikinci trainer |
| T-B-ZLIVE | §39 | `r_range == z_expected` ve `domain.z == z_expected` | `sort`/`unique`/`range(0.05,2.0)` / `_regulator_grid` |
| T-B-VEC | sample sonuç `D` | `length(vec(D))==length(z_expected)` | 1×N’i vektörleştirmeme |
| T-B-PRED203 | §39 | fırlatır; assess abort etmez; `:predict_threw`; `n_attempted==5` | sessiz sonlu `X`; tüm Q4 abort; 203 retry |
| T-B-NC | sahte `TrainingResult` `NotConverged` + sonlu D/traj | `included==true` | Success filtresi |
| T-B-MSG | `included=false` | neden+mesaj tutulur | boş mesaj |
| T-B-HP | beş fit-giriş kwargs | `(100,50,Symbol[],nothing)` | kaynak `for adam` yokluğu tek kanıt |
| T-B-HP-SENTINEL | §39 | `fit_config[k] == frozen_m3_training_config`; holdout-optimal `adam=50` canlıda kullanılmaz | gizli yardımcı seçimi; holdout önce |
| T-B-INC-HOLD | §39 | sonlu D+X + kötü holdout ⇒ `included==true` | `included = holdout_error < threshold` |
| T-B-HP-SRC | `FunctionalIdentifiability.jl` kaynağı | `evaluate_holdout` / holdout residual karar yok | karar yolunda holdout |

Sahte `TrainingResult` **sayısal olarak farklı** `params.nn`
taşımak zorundadır. Aynı nesne / `deepcopy` casusu düşer.

**Rollback:** B gövdesini ve M3 dikişlerini kaldır; A tipleri
kalabilir. M2 observer imzaları geri bozulmaz.

**Kabul:** Beş fit-giriş kaydı; `p0` izleri bağımsız
`build_ude_model` ile eşleşir; her sample params fit
sonucuyla bağlanır; her sample `r_range==z_expected`;
203 yalıtılır; HP beş restart’ta donmuş; kötü holdout
dahil etmeyi bozmaz.

**Bağımlılık:** M3-A.

---

### M3-C — Tanı montajı

**Dosyalar**

- `src/FunctionalIdentifiability.jl` —
  `assess_functional_identifiability` owner kapanışı:
  muhasebe, `binomial` çiftler, türetilmiş medyan/bayrak/status
- `test/test_functional_identifiability.jl` — C testleri

**Sorumluluklar**

- L-OWNER, L-ACCT, L-PAIR, L-DBIND, L-INDEP (canlı montaj),
  L-DERIVE, L-STAT, L-ZIMM (2 vs 5 başarı aynı `z`)
- T-C-DBIND, T-C-DSOURCE, T-C-TBIND, T-C-DERIVE-LIVE,
  T-C-ZERO-LIVE
- canlı sampled \(D = vec(D)\) ve canlı dönen `X` ile çift
  metrik bağlama

**Dışında**

- format (D)
- Q1/Q3/Q5/Q7 sahipliği
- `MechanismRecoveryResult` alanı
- truth \(D\)
- `equation_to_function` / `normalize_destruction_samples`
  birincil \(D\)
- yalnız kurucu A/B/C ile T-C-DERIVE-LIVE’ı kapatmak

**Testler**

| ID | Canlı gözlem | Bağımsız beklenti | Kırmızı gövde |
|---|---|---|---|
| T-C-ACCT | assess dönüşü | `n_attempted==5`; `n_successful==count(included)`; `n_failed==n_attempted-n_successful` | şişkin başarı |
| T-C-LEN | `restart_seeds` uzunluk ≠5 | `ArgumentError` | `m=3` sessiz complete |
| T-C-COMP | montaj | `complete===(n_attempted==5 && n_successful≥3)` | 2 başarı complete |
| T-C-BIN | `pairs` | `length(pairs)==binomial(n_successful,2)` | kötü çifti düş |
| T-C-IJ | her çift | `seed_i < seed_j`; self yok | `(201,201)`; `(j,i)` mükerrer |
| T-C-ZSAME | her çift | aynı `domain.z` | çifte özel domain |
| T-C-ZIMM | 2 included / 5 included | aynı `z` | başarıya göre domain |
| T-C-DBIND | §39 | `pair.* ==` bağımsız `dot`/`rate_rel_rmse` (canlı `D`) | sabit çift; örneklemeden önce metrik |
| T-C-DSOURCE | §39 | çift kaynağı canlı sample çıktısı | `equation_to_function`; `normalize_destruction_samples`; truth \(D\) |
| T-C-TBIND | §39 | `pair.traj_rmse_*` bağımsız canlı `X` | uydurma / truth / kopya / sabit yörünge |
| T-C-DERIVE-LIVE | §39 | canlı `pairs`’ten türetilmiş bayrak/status; A/B/C `assess` üzerinden | `function_disagree=false` iken median ≥0.20; `function_disagree=trajectory_agree`; `:structurally_identifiable` |
| T-C-ZERO-LIVE | §39 | canlı `D_j=0` ⇒ `alpha` ve scale-norm `NaN`; çift durur | `eps`; `α=1`; çifti düş; sonlu başarılı |
| T-C-ABC | assemble A/B/C (yardımcı) | §24 | yörünge vekili — T-C-DERIVE-LIVE olmadan yetmez |
| T-C-FLAG-REJECT | tutarsız bayrak girişi | red | kurucuya serbest Boolean |
| T-C-FLAG | `!complete` | üç bayrak false | incomplete + true bayrak |
| T-C-STAT | beş status kuralı | L-STAT | `:certified` |

**Rollback:** `assess` montajını kaldır; B yalıtımı kalabilir.

**Kabul:** Muhasebe, `binomial` çiftler, L-DBIND, canlı `X`
bağı ve türetilmiş status kilitli. Serbest bayrak yazımı
imkânsız. Kurucu yeşili, canlı `assess` uydurmasını örtmez.

**Bağımlılık:** M3-B.

---

### M3-D — Format

**Dosyalar**

- `src/FunctionalIdentifiability.jl` —
  `format_functional_identifiability_diagnostic`,
  `format_q3_q4_side_by_side`
- `test/test_functional_identifiability.jl` — format fixture

**Sorumluluklar**

- L-FMT, L-FAILMSG, L-NEG
- Q3 nesnesi ayrı basılır; Q4 alanına karışmaz
- her restart, her sırasız çift, her çift \(D\) metriği, her
  çift yörünge, başarısızlık tohumu, neden, mesaj, medyanlar,
  status, complete

**Dışında**

- `format_protocol_result` değişikliği
- unique-claim stdout
- `report_recovery` çağrısı
- yalnız-medyan özet
- yüksek hatalı başarılı çift filtresi

**Testler**

| ID | Canlı / kaynak | Bağımsız beklenti | Kırmızı gövde |
|---|---|---|---|
| T-D-ALL | format çıktısı | 5 tohum + tüm çiftler + ölçek-norm \(D\) + yörünge + medyan + status + complete | yalnız medyan |
| T-D-FAIL | başarısız satır | tohum + `failure_reason` + `message` | mesajsız / restart silme |
| T-D-FIVE | beş status fixture | her status basılır | yalnız `:function_agree` |
| T-D-BAN | string tarama | yasak cümle yok | “functionally identifiable” / “certified” / “Bayesian credible” |
| T-D-MUST | string tarama | zorunlu cümleler var | “certificate” ima |
| T-D-SRC | `report_recovery` / `format_protocol_result` | Q4 çağrısı yok | stdout kaçak |
| T-D-NOFILT | yüksek hatalı başarılı çift | formatta durur | sessiz filtre |

**Rollback:** format fonksiyonlarını sil; tanı durur.

**Kabul:** Medyan-only çıktı imkânsız; negatifler ve her çift
görünür.

**Bağımlılık:** M3-C.

---

### M3-E — Adversarial / runtime testler

**Dosyalar**

- `test/test_functional_identifiability.jl` — tam katalog
  (A–D testleri burada toplanır; eksik L-* kapanır)
- `test/runtests.jl` — ucuz include (zaten A’da)
- format listesi (`.github/workflows/ci.yml` files[] ve/veya
  `test/test_software_hygiene.jl`) — **yalnız yeni dosya
  hijyeni**; pahalı job yok
- `test/test_holdout.jl`, `test/test_recovery_pipeline.jl` —
  L-M2INT doğrulaması (satır silinmez)

**Sorumluluklar**

- L-STRENGTH’i kapat: her L-* için canlı gözlem + bağımsız
  beklenti + kırmızı gövde
- T-E-WALK, T-E-M2HASH
- L-COST kaynak taraması
- L-LIVE zinciri tek testte
- L-TRUTH: `assess` imzası / çağrıda truth yok
- L-M2INT: M2-G1, M2-G2 ve §6 kilit ID’leri korunur
- `FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)`
- `functional_identifiability ∉ fieldnames(MechanismRecoveryResult)`

**Dışında**

- `test/run_recovery_hard.jl` Q4 assert
- 5× protokol UDE
- M2 kapı gevşetme
- M2 testlerini `isdefined` ile değiştirmek
- T-E-WALK’ı yalnız dosya-token taraması yapmak

**Testler**

| ID | Canlı / kaynak | Bağımsız beklenti | Kırmızı gövde |
|---|---|---|---|
| T-E-LIVE | assess→build→fit→sample→predict (fonksiyon içi) | dört adım da gözlenir | ölü kod; yardımcı-only |
| T-E-WALK | §39 | geçişli graf yasak hedeflere yol vermez | gizli yardımcı `discover_*` / `evaluate_holdout` / `_regulator_grid` |
| T-E-NOM4 | `FunctionalIdentifiability.jl` token | T-E-WALK ile birlikte; tek başına yetmez | dosya temiz, yardımcı kirli |
| T-E-COST | hard / ci / runtests | pahalı `assess` çağrısı yok | PR’a 5 UDE |
| T-E-THRESH | Q4 dosyası | `RECOVERY_THRESHOLDS` okunmaz | eşik kaydırma |
| T-E-HOLD | `unique_claim_kpis_hold` | Q4’süz aynı üç konjonkt | dördüncü konjonkt |
| T-E-TRUTH | `assess` methods | `truth_rate` / `D_true` yok | fixture cevabı |
| T-E-PROJ | Project.toml | StructuralIdentifiability / Turing / MCMC yok | paket kaçak |
| T-E-M2HASH | §39 | tüm M2 kilit ID’leri durur; `∉ names`; alan yok | M2 test silme; `!isdefined`; Q4’ü MRR’ye eklemek |
| T-E-CUT | cutoff | `FUNCTIONAL_ID_REPORTING_CUTOFFS ===` kilitli tuple | değer kaydırma |

**Rollback:** test dosyasını / include’u kaldır.

**Kabul:** §33 tablosundaki her saldırı yazılı teste bağlı.
§39 LIVE-path testleri yeşil. M2 test ailesi zayıflamamış.

**Bağımlılık:** M3-A…D. G script testi G ile birlikte kilitlenir.

---

### M3-F — Fisher terminolojisi

**Dosyalar**

- `src/Identifiability.jl` — `IdentifiabilityReport` docstring,
  `parameter_credible_intervals` docstring, `_z_score` mesajı
- `docs/src/identifiability-product.md` — “credible” kullanıcı
  cümlesi yok (H ile örtüşebilir; F dilin sahibi)
- `test/test_functional_identifiability.jl` veya mevcut honesty
  — L-FISH kaynak testi
- `test/test_identifiability_product.jl` yalnız **kırılan
  string** varsa retarget; semantik değişmez

**Sorumluluklar**

- L-FISH
- kullanıcı yüzünde “asymptotic Fisher interval” ve
  “nominal coverage”
- `parameter_credible_intervals` kullanıcı metni retarget
  (unexported ad kırılmak zorunda değil)
- `production_destruction_tradeoff` anahtarları durur
- yeni UQ / interval alanı yok

**Dışında**

- zorunlu rename
- Bayes implementasyonu
- Q3 semantiğini Q4 yapmak
- “credible interval” / “credible level” / Bayesian dil

**Testler**

| ID | Canlı / kaynak | Bağımsız beklenti | Kırmızı gövde |
|---|---|---|---|
| T-F-SRC | `src/Identifiability.jl` kullanıcı metni | “Bayesian” / “credible interval” / “credible level” yok | eski docstring / `_z_score` |
| T-F-NOM | aynı dosya | “asymptotic Fisher interval” + “nominal coverage” var | sessiz bırakma |
| T-F-Z | `_z_score` | “nominal coverage” (veya eşdeğer); “credible level” yok | eski `ArgumentError` |
| T-F-KEYS | tradeoff anahtarları | aynı | Q3/Q4 karışımı |
| T-F-PCI | `parameter_credible_intervals` docstring | asymptotic Fisher / nominal coverage | “Asymptotic normal credible intervals” |

**Rollback:** docstring’leri geri al.

**Kabul:** Kullanıcı yüzü Bayes ima etmez.

**Bağımlılık:** M3-A (test evrakı). Semantik olarak A–E’den bağımsız.

---

### M3-G — Gerçek benchmark script

**Dosyalar**

- yeni `benchmark/functional_identifiability.jl`
- `test/test_functional_identifiability.jl` — T-G-CALL
- isteğe bağlı ClaimScopeHonesty satırı (PR job değil)

**Sorumluluklar**

- L-BENCH, T-G-CALL
- script **gerçekten** `assess_functional_identifiability(...)`
  çağırır ve dönen tanıyı kullanır
- Hill `m=5` protokol kwargs (zorunlu `tspan`/`n_points`/`noise_σ`)
- tanıyı imal etmez
- benchmark gerçektir; uydurma değildir

**Dışında**

- GitHub nightly job (M7)
- PR recovery’ye ekleme
- MM’yi ikinci iddia yazma
- `FunctionalIdentifiabilityDiagnostic(...)` ile sahte tablo
- beş-restart gerçek UDE’yi PR testine eklemek
- yalnız string `contains`

**Testler**

| ID | Canlı / kaynak | Bağımsız beklenti | Kırmızı gövde |
|---|---|---|---|
| T-G-CALL | §39; script **çalıştırılır**; gerçek çağrı sayacı | assess giriş casusu ≥1; dönen tanı formatlanır | doğrudan `FunctionalIdentifiabilityDiagnostic(...)`; sahte tanı; yorum; kullanılmayan referans |
| T-G-AST | script AST | `assess_functional_identifiability` bir **Call** düğümüdür | comment / unused |
| T-G-FMT | script format çağrısı | casusun döndürdüğü tanı kullanılır | medyan uydurma |
| T-G-KW | `generate_recovery_experiments` | `tspan`/`n_points`/`noise_σ` | `protocol=` |
| T-G-CI | `ci.yml` | script recovery/test job’unda yok | PR maliyeti |

**Rollback:** script’i sil; A–F durur. Complete geri çekilir.

**Kabul:** Script canlı `assess` yoludur. Nightly job olmasa da
script vardır ve `assess`’i çağırır. M3 complete script’siz
veya uydurma tanı ile ilan edilemez.

**Bağımlılık:** M3-C, M3-D.

---

### M3-H — Doküman / sözleşme retarget

**Dosyalar**

- `docs/src/design/v1_contract.md` — Q4:
  `implemented as a practical diagnostic, not a gate`;
  “Q4 is not a formal identifiability certificate” durur
- `docs/src/identifiability-product.md` — Q3 vs Q4
- `docs/src/unique-claim.md`, `docs/src/stability.md`,
  `docs/src/out-of-scope.md`, `docs/src/index.md`,
  `docs/src/architecture.md`, `README.md`, `CHANGELOG.md` —
  Q4 tanı; hold hâlâ Q3+Q1+Q5
- `docs/research/V1-IMPLEMENTATION-PLAN.md` — M3 bölümü
  (bu planla uyumlu; bu turda zaten retarget edilir)
- `test/test_v1_contract.jl` — “not implemented” → tanı var,
  kapı değil
- `test/test_docs_honesty.jl` — gerekirse cümle kilitleri
- M3-A’da yapılmamışsa `∉ names` retarget

**Sorumluluklar**

- L-M2RET / L-M2INT (kalan sözleşme)
- kullanıcı cümleleri L-FMT ile aynı yasak/zorunlu set
- FORENSIC-AUDIT “artık var, sertifika değil” **M8’e
  bırakılabilir**

**Dışında**

- hold değişimi
- public export
- vizyona “v1.0 implemented” yazmak
- M4/M5 dokümanı
- M2 testlerini zayıflatmak
- Q4’ü `MechanismRecoveryResult`’a eklemek

**Testler**

| ID | Canlı / kaynak | Bağımsız beklenti | Kırmızı gövde |
|---|---|---|---|
| T-H-CERT | sözleşme / landing | certificate cümlesi durur | “functionally identifiable” |
| T-H-GATE | “not a gate” | Q4 hold konjonkti yok | hold’a Q4 ekleme |
| T-H-M2 | `functional_identifiability ∉ fieldnames(MechanismRecoveryResult)` | alan yok | alan ekleme |
| T-H-NAMES | `FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)` | unexported | `!isdefined` geri dönüş |
| T-H-Q7 | Q7 “not implemented” değildir | Q7 durur | Q7 gerilemesi |

**Rollback:** sözleşme Q4’ü `not implemented`e döner; kod
include’u da geri alınır.

**Kabul:** Doküman ve `test_v1_contract` aynı cümleyi söyler.

**Bağımlılık:** M3-A include + C tanı. F dili ile uyumlu.

---

## 35. Uygulama sırası

```
M3-A  domain + metrik + ∉ names retarget + include + türetilmiş bayrak
  → M3-B  canlı restart + p0/params/z/predict casusu + HP/inclusion sentinel
    → M3-C  montaj + LIVE D/X bağ + derive-live + zero-live
      → M3-D  format
        → M3-E  walk + M2 hash + adversarial kapanış
M3-F  Fisher dili          (A sonrası paralel olabilir)
M3-G  gerçek script CALL   (C+D sonrası)
M3-H  sözleşme / landing   (A include + C sonrası)
```

`include` ve `∉ names` retarget **aynı adımda** (A).

Plan Mode sonrası Agent. Dilimler küçük ve geri alınabilir.

---

## 36. Global rollback

1. `BioDynaX.jl` include’unu kaldır
2. `test/runtests.jl` include’unu kaldır
3. `src/FunctionalIdentifiability.jl` ve
   `benchmark/functional_identifiability.jl` ve
   `test/test_functional_identifiability.jl` sil
4. M3 fit-giriş / sample-sonuç / `predict_ude` dikişlerini kaldır
   (M2 observer imzaları durur)
5. `∉ names` retarget’i geri al **yalnız** tip de silinirse
6. `Identifiability.jl` dilini geri al
7. v1_contract Q4 `not implemented`e döner

`fit_unknown_destruction` imzası değişmemişse unique-claim
bit-özdeş kalır. Değişmişse `seed=0` default M2’yi korur.

---

## 37. Non-goals

- M4 occupancy keşif / dummy-time kaldırma / graph-local
  eğitilmiş \(D\)
- M5 baseline
- `MechanismRecoveryResult.functional_identifiability`
- public `FunctionalIdentifiabilityDiagnostic`
- Q4’ün `unique_claim_kpis_hold` / hard kapıya girmesi
- `RECOVERY_THRESHOLDS` gevşetme veya sıkılaştırma
- her PR’da 5× full UDE
- Bayes / OED / StructuralIdentifiability.jl
- hipotez ranking
- kanonik Hill-from-NN
- vizyon \(\dot x = f+D\)
- `validate_network` tek-delik kapısı
- suite RNG bit-özdeş generate zorunluluğu
- M4 tohum listesini Q4 saymak
- ikinci recovery pipeline
- `DestructionSamples`
- Q4 `success` / `passed` kapısı
- M2 7/2, tek generate, train-only fit, train-türevli keşif
  domain, holdout evaluator sahipliği, M2 \(D\) metrikleri,
  M2 residual toplama, Q7 semantiği, `0.30` semantiği, M2
  eşikleri, M2 public export, M2 stdout blokları

---

## 38. M3 complete kabulü

M3 ancak **hepsi** doğruysa complete’tir:

1. M3-A…H uygulandı
2. L-STRENGTH: kritik her değişmez canlı fonksiyon içi gözlem +
   bağımsız beklenti + karşılaştırma + somut kırmızı gövde
3. §39’daki on dört LIVE-path testi yazılı ve kırmızı gövdeli
4. `live_p0_fingerprint[k] == fingerprint(build_ude_model(
   MersenneTwister(FUNCTIONAL_ID_RESTART_SEEDS[k]), ude_net)[2].nn)`
5. `live_sample_params_fingerprint[k] == fingerprint(fit_result[k].params.nn)`
   ve `live_D[k] == independently_sampled_D[k]`; beş ayrı final
   params ⇒ `length(unique(live_D_fingerprints)) == n_successful`
6. Her `sample_unknown_destruction_grid` `r_range == z_expected`
   ve `diagnostic.domain.z == z_expected` (sentinel
   `[0.1, 0.5, 0.1, 0.8]`)
7. Çift \(D\) metrikleri canlı `D = vec(D)` ile bağımsız yeniden
   hesaplanır; `equation_to_function` /
   `normalize_destruction_samples` birincil \(D\) değildir
8. Çift yörünge metrikleri canlı dönen `X` ile bağımsız yeniden
   hesaplanır
9. 203 `predict_ude` fırlatması yalıtılır; `n_attempted==5`;
   retry yok; fırlatma sonlu çıktıya çevrilmez
10. `function_disagree` / `trajectory_agree` / `status` canlı
    `assess.pairs`’ten türetilir; A/B/C `assess` üzerinden
11. Canlı `D_j=0` ⇒ `alpha` ve scale-norm `NaN`; çift korunur
12. `n_attempted==5`, `n_successful==count(included)`,
    `n_failed==n_attempted-n_successful`,
    `complete===(n_attempted==5 && n_successful>=3)`,
    `length(pairs)==binomial(n_successful,2)`, `seed_i < seed_j`
13. `NotConverged` + sonlu \(D\)/yörünge otomatik dışlama değildir
14. Sonlu D+X + kasten kötü holdout ⇒ `included==true`
15. Beş fit `fit_config[k] == (100, 50, Symbol[], nothing)`;
    holdout-optimal `adam=50` canlıda yoktur; holdout beş fit
    kararından önce çağrılmaz
16. T-E-WALK geçişli yasak yol yoktur
17. Beş status + her restart + her çift + başarısız mesaj formatta
18. `benchmark/functional_identifiability.jl` `assess`’i **çağırır**
    (gerçek sayacı) ve dönen tanıyı kullanır
19. PR / hard / `runtests` 5× UDE çalıştırmaz
20. M2 7/2, hold, 0.30, `evaluate_holdout` sahipliği,
    M2-G1/G2 ve L-* testleri (T-E-M2HASH),
    `FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)`,
    `functional_identifiability ∉ fieldnames(MechanismRecoveryResult)`
    durur
21. Fisher kullanıcı dili asymptotic Fisher interval / nominal
    coverage; `parameter_credible_intervals` metni retarget
22. Hiçbir satır “functionally identifiable” / “certificate”
    / “Bayesian credible” demez
23. Cutoff tuple `===` kilitli; `RECOVERY_THRESHOLDS` okunmaz
24. Birim test yeşili tek başına yetmez

Nightly GitHub job M7’dir. Script’in **çağrı davranışı** M3
complete’in parçasıdır; çizelgeleme değildir.

---

## 39. Zorunlu LIVE-path katalog

Aşağıdaki testler **LIVE-path** testleridir. Yardımcı testi
değildir. `assess_functional_identifiability` yardımcıyı
atlayıp sonucu uydurabiliyorsa test yeşil **kalamaz**.

Her kayıt: gözlem yeri, bağımsız beklenen değer, somut yanlış
implementasyon, beklenen kırmızı davranış.

### T-B-P0 — canlı eğitim `p0` bağ

**Gözlem yeri.** `fit_unknown_destruction` **içi**, eğitimden
önce, canlı `assess` çağrısı sırasında. Her restart için:
tohum sırası `k`, gerçek `p0` sayısal parmak izi, gerçek
`ExperimentSet` kimliği, gerçek `length(set)`, gerçek eğitim
kwargs.

**Bağımsız beklenen.**

```
live_p0_fingerprint[k] ==
    fingerprint(
        build_ude_model(
            MersenneTwister(FUNCTIONAL_ID_RESTART_SEEDS[k]),
            ude_net
        )[2].nn
    )
```

**Yanlış implementasyon.** Tek paylaşılan `p0`; sabit RNG;
yalnız seed etiketi; seed 103 init’inin `201..205` etiketi
altında reuse; metadata değişip init özdeş.

**Kırmızı.** Herhangi bir `k` için parmak izi eşleşmez.
Beş etiket, tek `p0` izi.

**Not.** T-B-P0, tek gerçek fit + beş `deepcopy(final params)`
saldırısını tek başına öldürmez. T-B-PARAMS zorunludur.

### T-B-PARAMS — canlı fit params → sampler → D

**Gözlem yeri.** (1) Restart `k`’nin gerçek
`TrainingResult.params`. (2) Aynı restart’ın canlı
`sample_unknown_destruction_grid` çağrısının **içi**:
kullanılan `params` ve dönen `D`.

**Bağımsız beklenen.** Test, `fit_result[k].params` ve
bağımsız `z_expected` ile
`sample_unknown_destruction_grid`’i kendisi çağırır.

```
live_sample_params_fingerprint[k] ==
    fingerprint(fit_result[k].params.nn)

live_D[k] == independently_sampled_D[k]
```

Casus beş sayısal olarak farklı final params verdiğinde:

```
length(unique(live_D_fingerprints)) == n_successful
```

**Yanlış implementasyon.**

```
# bir gerçek fit; final params beş kez kopya
params = fit_once.params
for k in 1:5
    use deepcopy(params)          # veya aynı nesne
    label seed = 200+k
end

# farklı fit sonuçları, sampler paylaşılan params
# farklı fit sonuçları, aynı sampled D
```

`===` nesne kimliği tek kanıt **değildir**.

**Kırmızı.** Herhangi bir `k` için sample params izi fit
sonucuyla eşleşmez; veya canlı \(D_k\) bağımsız yeniden
örnekle eşleşmez; veya beş ayrı params’a rağmen tek \(D\)
izi.

### T-B-HP-SENTINEL — holdout HP seçemez (sıra + decoy)

**Gözlem yeri.** Fit-giriş casusu, **herhangi bir holdout
değerlendirmesinden önce** kayıtlı `adam` / `bfgs` /
`frozen_phys` / `phys_init`. Çağrı-sıra günlüğü: beş
fit-giriş, ilk holdout residual / `evaluate_holdout`
olayından önce.

**Bağımsız beklenen.**

```
frozen_m3_training_config = (100, 50, Symbol[], nothing)
fit_config[k] == frozen_m3_training_config    # her k
```

Sentinel, `adam = 50` konfigürasyonunu holdout metriğinde
daha iyi yapar. Canlı fit yine `adam = 100` alır.
Beş fit kararından önce holdout residual /
`evaluate_holdout` yoktur.

**Yanlış implementasyon.**

```
for adam in [50, 100, 500]
    fit(...; adam)
    choose_using_holdout(...)
end

_select_training_by_holdout(split.holdout)  # gizli yardımcı
```

Kaynakta `for adam in` yokluğu yeterli **değildir**.

**Kırmızı.** Herhangi bir restart `adam ≠ 100` (veya diğer
donmuş alan sapması) görür; veya holdout değerlendirmesi
beş fit’ten önce damgalanır; veya gizli yardımcı seçim
yapar.

### T-B-INC-HOLD — holdout dahil etmeyi seçemez

**Gözlem yeri.** Canlı `assess` restart kaydı. Fikstür:
sonlu \(D\), sonlu yörünge (`predict_ude` sonlu `X`),
kasten kötü holdout metriği (`≥ 1e3` veya belgelenmiş
eşiğin açıkça üstü).

**Bağımsız beklenen.**

```
included == true
```

Dahil etme belgelenmiş M3 kriterleridir (sonlu D, sonlu
yörünge, `failure_reason == :none`). Holdout kalitesi
değildir. Holdout değerlendirmesi beş eğitim + dahil etme
kararından sonradır.

**Yanlış implementasyon.**

```
included = holdout_error < threshold
included = evaluate_holdout(...).residual < 0.30
included = traj_rmse_holdout < 0.05
```

**Kırmızı.** Sentinel restart `included == false`. Tesadüfen
iyi holdout ile yeşil kalmak kabul edilmez; holdout kasten
kötü olmak **zorundadır**.

### T-B-PRED203 — tohum 203 prediction yalıtımı

**Gözlem yeri.** `predict_ude` **içi** (203’ün params izine
bağlı fırlatma) ve canlı `assess` dönüşü.

**Bağımsız beklenen.** `predict_ude` fırlatır; `assess`
fırlatmaz; 203 `restarts` içindedir; `included == false`;
`failure_reason == :predict_threw`; `message != ""`;
`n_attempted == 5`; `attempt_count[203] == 1`; diğer
restart’lar devam eder.

**Yanlış implementasyon.** Tüm Q4 abort; fırlatmayı sonlu
`X`’e çevirmek; 203’ü silmek; 203’ü yeniden denemek;
sessiz kabul.

**Kırmızı.** Yukarıdaki beklenenlerden herhangi biri
bozulur.

### T-B-ZLIVE — canlı domain `z`

**Gözlem yeri.** Canlı her `sample_unknown_destruction_grid`
çağrısının `r_range`’i ve `diagnostic.domain.z`.

**Bağımsız beklenen.** Sentinel gözlemler
`[0.1, 0.5, 0.1, 0.8]`. Test kendi kurar:

```
z_expected = vcat(train_regulator_observations,
                  holdout_regulator_observations)
```

Sıra, tekrarlar ve tam değerler korunur.

```
r_range == z_expected
diagnostic.domain.z == z_expected
```

**Yanlış implementasyon.** `sort`; `unique`;
`_regulator_grid`; `_unique_claim_external_regulator_band`;
`range(0.05, 2.0)`; restart-türevli / params-türevli /
\(D\)-türevli domain; post-hoc kırpma; `construction`
yalanı.

**Kırmızı.** `r_range` veya `domain.z` `z_expected` değil.

### T-C-DBIND — canlı çift \(D\) bağ

**Gözlem yeri.** L-SAMPSPY `(R, D, term)`; `D = vec(D)`;
canlı `assess.pairs`.

**Bağımsız beklenen.** Canlı \(D_i, D_j\) üzerinde testin
kendi `dot` / `rate_rel_rmse` hesabı `pair.*` ile eşleşir.

**Yanlış implementasyon.** Örneklemeden önce metrik; sabit
çift; sampler’dan bağımsız sayı.

**Kırmızı.** Herhangi bir çift alanı bağımsız yeniden
hesapla eşleşmez.

### T-C-DSOURCE — canlı \(D\) kaynağı

**Gözlem yeri.** Canlı `sample_unknown_destruction_grid`
çıktısı ile aynı `assess` çağrısının `pairs` alanları.

**Bağımsız beklenen.** Çift \(D\) metriklerinin tek kaynağı
o canlı sample çıktısıdır. T-B-PARAMS’taki bağımsız yeniden
örnekle tutarlıdır.

**Yanlış implementasyon.** `equation_to_function`;
`normalize_destruction_samples`; sembolik \(D\); truth
\(D\); keyfi \(D\); `assess`’in sample sonrası başka \(D\)
uydurması.

**Kırmızı.** Çift metrikleri canlı sample \(D\)’sinden
türetilmez.

### T-C-TBIND — canlı yörünge `X` bağ

**Gözlem yeri.** `predict_ude` **içi**, dönen `X` (girdi
`params` değil). Her dahil restart, her train/holdout
deneyi.

**Bağımsız beklenen.**

```
bağımsız traj = mean(rate_rel_rmse(vec(X_i), vec(X_j)))
pair.traj_rmse_train   == bağımsız train
pair.traj_rmse_holdout == bağımsız holdout
```

**Yanlış implementasyon.** Uydurma yörünge; truth’tan
metrik; başka restart’tan kopya; sabit sayı; yalnız girdi
params casusu.

**Kırmızı.** `pair.traj_rmse_*` canlı `X`’ten bağımsız
yeniden hesapla eşleşmez.

### T-C-DERIVE-LIVE — canlı bayrak / status türetimi

**Gözlem yeri.** Canlı `assess_functional_identifiability`
dönüşü — özellikle `pairs`, `complete`, `n_successful`,
bayraklar, `status`. Kurucu / `assemble` değil.

Test canlı `pairs`’ten bağımsız hesaplar:

```
complete
n_successful
median(d_rmse_scale_normalized)
median trajectory metric
```

**Bağımsız beklenen.**

```
function_disagree === (
    complete &&
    n_successful >= 2 &&
    median(d_rmse_scale_normalized) >= 0.20
)
```

(dokümante incomplete/failure hâli yorumu engellemiyorsa).
`trajectory_agree` belgelenmiş `0.05` kesiti ve canlı çift
metriklerinden. `status` complete / yörünge / fonksiyon
kurallarından. `assess` bu alanlar için bağımsız Boolean
**almaz**.

Canlı A/B/C fikstürü **`assess` üzerinden**:

```
A: traj near, D near  ⇒ function_disagree = false
B: traj near, D far   ⇒ function_disagree = true
                      ⇒ trajectory_agree_function_disagree = true
C: traj far,  D far   ⇒ trajectory_agree = false
                      ⇒ function_disagree = true
```

**Yanlış implementasyon.**

```
function_disagree = false          # median ≥ 0.20 iken
function_disagree = trajectory_agree
status = :structurally_identifiable
```

**Kırmızı.** Canlı `pairs`’ten türetilen değer ile `assess`
alanı çelişir; veya A/B/C hücreleri `assess` üzerinde
ayrışmaz.

### T-C-ZERO-LIVE — canlı sıfır payda

**Gözlem yeri.** Canlı `assess` çifti; bir restart’ın canlı
örneklenen \(D_j = [0,0,\ldots,0]\).

**Bağımsız beklenen.**

```
alpha = NaN
d_rmse_scale_normalized = NaN
```

Çift `pairs` içinde durur.

**Yanlış implementasyon.** `eps` payda; `alpha = 1`; çifti
silmek; sonlu metrik ile başarılı işaretlemek.

**Kırmızı.** `alpha` veya scale-norm sonlu; veya çift yok;
veya çift “başarılı sonlu” görünür. T-A-ZERO yeşili bu
kırmızıyı örtmez.

### T-E-WALK — gizli keşif yürüyüşü

**Gözlem yeri.** `assess_functional_identifiability`
girişinden, `FunctionalIdentifiability.jl` ve ondan
erişilebilir yerel yardımcıların **geçişli** çağrı-grafı
(M2-G1 / L-DISC-B-2 standardı).

**Bağımsız beklenen.** Yasak hedeflere yol yoktur:
`discover_unknown_rate`, `discover_equations`,
`discover_unknown_destruction`, `run_recovery_suite`,
`_train_unknown_edge`, `_evaluate_unknown_rate_recovery`,
`evaluate_holdout`, `report_recovery`, `_regulator_grid`,
`_unique_claim_external_regulator_band`,
`range(0.05, 2.0`, `range(0.0, 1.0`.

**Yanlış implementasyon.** Gizli yardımcı; çok-düzey
dolayım; dosya token’ı temiz / yardımcı kirli.

**Kırmızı.** Geçişli graf yasak hedefe bir kenar içerir.
Bir-düzey tarama yetmez.

### T-G-CALL — benchmark gerçekten `assess` çağırır

**Gözlem yeri.** `benchmark/functional_identifiability.jl`
M3 casusları altında **çalıştırılır**.
`assess_functional_identifiability` giriş casusu / çağrı
sayacı.

**Bağımsız beklenen.** Sayacı ≥ 1. Dönen nesne formatın
tükettiği tanıdır.

**Yanlış implementasyon.** Yorum; string; kullanılmayan
referans; doğrudan `FunctionalIdentifiabilityDiagnostic(...)`;
sahte benchmark çıktısı; `assess`’i çalıştırmayan sarmalayıcı.

**Kırmızı.** Sayacı 0; veya format edilen nesne `assess`
dönüşü değildir.

### T-E-M2HASH — M2 test bütünlüğü

**Gözlem yeri.** Canlı M2 test dosyaları
(`test/test_holdout.jl`, `test/test_recovery_pipeline.jl`
ve M2-G1/G2 kilitlerinin bulunduğu diğer dosyalar).
`@testset` adları ve iki zorunlu assert.

**Bağımsız beklenen.** Şu kilit ID’lerinin her biri durur:

```
L-SPLIT-ID
L-SET-INTACT
L-FIT-A
L-FIT-B
L-RNG
L-DOM-A
L-DOM-B
L-D-OCC
L-OVERFIT
L-RES-HOLD
L-RES-LEGACY
L-DISC-A
L-DISC-B-1
L-DISC-B-2
L-DISC-B-3
L-EARLY
L-GATE
L-SITE
L-API
L-M34
```

Ayrıca:

```
FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
functional_identifiability ∉ fieldnames(MechanismRecoveryResult)
```

`!isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)`
Q4 yokluğu için **kullanılmaz**.

**Yanlış implementasyon.** M2-G1/G2 veya L-* testini silmek
veya gevşetmek; `!isdefined` geri dönüşü; Q4’ü
`MechanismRecoveryResult`’a eklemek.

**Kırmızı.** Herhangi bir kilit ID kaybolur; veya Q4 yokluğu
`!isdefined` ile yazılır; veya
`functional_identifiability ∈ fieldnames(MechanismRecoveryResult)`.

---

## 40. Bu belgenin kapsamı

Bu turda uygulanmaz. Kaynak, test, benchmark ve CI
değiştirilmez. Commit / push yoktur.
