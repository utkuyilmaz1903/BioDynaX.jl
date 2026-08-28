---
name: BioDynaX v1.0 Plan
overview: "Mevcut 0.9.2 one-hole iş akışını vizyonun tamamını uygulamadan araştırma kalitesine taşımak: önce bilimsel sözleşmeyi sabitlemek, recovery yığınını ayrıştırmak, held-out ve fonksiyonel identifiability eklemek, sonra tohum/graph-local/baseline ve yeniden üretilebilirlik kapılarını güçlendirmek."
todos:
  - id: m0-contract
    content: "M0: docs/src/design/v1_contract.md — P-D·u, Q1–Q7, kapalı iddialar; vizyon denklemini ürün yapma"
    status: completed
  - id: m1-pipeline
    content: "M1: run_recovery_suite ayrıştır + MechanismRecoveryResult; honesty string silme yok"
    status: pending
  - id: m2-heldout
    content: "M2: ExperimentSplit + holdout residual ve D hatası; 0.30’u körlemesine sıkılaştırma"
    status: pending
  - id: m3-functional-id
    content: "M3: FunctionalIdentifiabilityDiagnostic; Q3 ölçek uyarısı vs Q4 fonksiyon tanısı; Fisher’a credible deme"
    status: pending
  - id: m4-robustness
    content: "M4: yörünge-örnekli D, eğitilmiş-UDE graph-local, çok tohum artifact (her PR’da N×40 dk yok)"
    status: pending
  - id: m5-baselines
    content: "M5: aynı görev saf UDE + ExplicitSTLSQ + ImplicitSINDyPI; DDS skip≠win"
    status: pending
  - id: m6-repro
    content: "M6: RunMetadata golden path + research Manifest + persist KPI artifact"
    status: pending
  - id: m7-package-ci
    content: "M7: katmanlı CI, format tüm src/test, Aqua extras, honesty büyütmeyi durdur"
    status: pending
  - id: m8-docs
    content: "M8: landing docs Q-katmanları ve dummy-time/fonksiyon-regresyon dürüstlüğü"
    status: pending
  - id: m9-perf-release
    content: "M9–M10: allocation bekçisi; bilimsel v1.0 tag; General/JOSS ayrı maintainer kapısı"
    status: pending
isProject: false
---

# BioDynaX v1.0 Geliştirme Planı

Bu plan, [docs/research/BIO-DYNAX-VISION.md](docs/research/BIO-DYNAX-VISION.md) ve [docs/research/FORENSIC-AUDIT.md](docs/research/FORENSIC-AUDIT.md) ile **gerçek kaynak ağacını** yan yana okuyarak yazıldı. Adli denetim büyük ölçüde doğru; birkaç yerde abartılı veya yanlış teşhis var. v1.0, Mechanism Repair motoru değildir. Hedef: bilinen graf + tam bir bilinmeyen yıkım + çoklu deney + mekanistik kurtarma + identifiability + bilimsel doğrulama + yeniden üretilebilirlik.

## Denetimin koda karşı doğrulanması

Aşağıdaki 12 zayıflık koda karşı kontrol edildi.

- **1. Fonksiyonel identifiability yok — DOĞRU.** [src/Identifiability.jl](src/Identifiability.jl) yalnızca fiziksel parametre Fisher’ı ve `k_prod`↔`D` ölçek kosinüsüdür. `D_1…D_m` karşılaştırması yok. [src/ScientificCore.jl](src/ScientificCore.jl) içindeki `DiscoveryUncertaintyReport` bootstrap terim sıklığıdır; fonksiyonel anlaşma değildir.
- **2. Residual / identifiability held-out değil — DOĞRU.** [src/Recovery.jl](src/Recovery.jl) `ref_exp = first(ude_set.experiments)` kullanır (satır ~1088–1101). Kullanıcı yüzü [examples/unknown_inhibition.jl](examples/unknown_inhibition.jl) de residual/ident’i `first_exp` üzerinde hesaplar. [src/Experiments.jl](src/Experiments.jl) `ExperimentSet` bir listedir; `experiment_batches` yalnızca Adam minibatch’tir. Discovery `validation_fraction` **sütun** bölmesidir, deney split’i değildir.
- **3. Tek tohuma bağımlılık — DOĞRU, nüanslı.** CI kırmızı kapısı seed 103’tür ([test/test_recovery_hard.jl](test/test_recovery_hard.jl)). [benchmark/recovery_seeds.jl](benchmark/recovery_seeds.jl) `--ude` raporu vardır ama kapı değildir ([docs/src/out-of-scope.md](docs/src/out-of-scope.md)).
- **4. Zayıf mekanistik metrikler — DOĞRU.** Kapılar: recall ≥ 0.99 (monomial üyelik), residual ≤ 0.30 (tipik ~0.003), F1 tabanı 0.50, `unidentifiable_edge == true`. Extras’lı `D` residual kapısını geçebilir.
- **5. Sentetik 1D + sahte zaman — DOĞRU, teşhis inceltilmeli.** Unique-claim `discover_unknown_rate` `times = range(0,1)` ve `derivatives = D_nn` verir ([src/Recovery.jl](src/Recovery.jl) ~675–680). Bu bilimsel olarak **fonksiyon regresyonudur**, `ẋ`-SINDy değildir. [src/Discovery.jl](src/Discovery.jl) içinde gerçek yörünge yolu (`_collect_trajectory_data`) vardır; unique-claim onu `D` için kullanmaz. Asıl boşluk sahte zamandan çok **ızgaranın eğitim yörüngesi occupancy’si olmamasıdır**.
- **6. Graph-local eğitilmiş UDE `D` üzerinde değil — DOĞRU.** Ablation analitik Hill + distractor `z` kullanır ([src/Recovery.jl](src/Recovery.jl) ~1142+). [test/test_graph_local_library.jl](test/test_graph_local_library.jl) çoğunlukla kütüphane üyeliği ve kaynak-string kilitleridir.
- **7. Harici baseline eksik — DOĞRU.** [benchmark/external_baseline.md](benchmark/external_baseline.md): DataDrivenSparse resolve olmaz. CI `continue-on-error`. İç kontrol graph vs global F1 eşitlenebilir; kilit kütüphane üyeliğidir.
- **8. Sert kodlu bilimsel alanlar — KISMEN DOĞRU.** `canonical_hill_from_nn = false` **bilinçli kapalı iddiadır**, sahte başarı değildir. `practical_not_structural = true` yöntem sınıfı sabitidir. Asıl sorun: `coefficients_are_biological_constants = !unidentifiable_edge` bağımsız ölçüm değildir; başarı `unidentifiable_edge == true` ister. Fisher aralıkları “credible interval” diye adlandırılır ([src/Identifiability.jl](src/Identifiability.jl) ~120–122) — bu yanlış güven dilidir.
- **9. Yeniden üretilebilirlik zayıf — DOĞRU.** Kök `Manifest.toml` yok (`.gitignore` `Manifest*.toml`). `RunMetadata` ([src/Types.jl](src/Types.jl)) golden path’e bağlanmaz. Benchmark artifact commit edilmez.
- **10. Şişmiş recovery / honesty — DOĞRU VE ALTINDA KALMIŞ.** [src/Recovery.jl](src/Recovery.jl) `run_recovery_suite` ~600 satırlık `Dict{Symbol,Any}` orkestratördür (varsayılan 7 + `:three_state`, `:wrong_graph`, `:identifiability`, `:partial_obs`, `:six_state`, …); RNG için dummy `build_ude_model` çağırır. Honesty yığını ~9k satır / ~19 `*_contract_holds()`; en büyüğü [src/ExperimentCheckpoint.jl](src/ExperimentCheckpoint.jl) (~1571). **Nüans:** [src/HybridCompose.jl](src/HybridCompose.jl) / [src/HybridResidual.jl](src/HybridResidual.jl) ikinci `compose_hybrid_rhs` üretimi değildir — `Recovery.jl` tek production, diğerleri sözleşme/fixture aynasıdır. `sample_learned_function` vs `sample_unknown_destruction` çift yol durur. Gizli `Ref` sayaçları: `COMPILE_NETWORK_COUNTER`, `TRAIN_UNKNOWN_EDGE_COUNTER`.
- **11. v1.0 sözleşmesi yok — DOĞRU.** `docs/src/design/` yok. Vizyon \(\dot x = f_{\mathrm{known}}+D\); kod \(P-D\cdot u\).
- **12. Paket / bilim kapıları ayrı — DOĞRU.** Format dosya listesidir ([.github/workflows/ci.yml](.github/workflows/ci.yml)). `standards` ayrı ve “kırmızı eşiği”dir. Recovery yalnızca Ubuntu × Julia 1.10. Aqua/JET `Project.toml` extras’ta değildir.

**Denetimin abarttığı / v1.0’a taşınmaması gerekenler**

- Yaş laboratuvar verisi v1.0 bilimsel bloğu değildir; sentetik metodoloji araştırma-grade olabilir. Yokluk kilidi kalsın.
- Derleyiciyi vizyon denklemine uydurmak. v1.0 \(P-D\cdot u\)’yu sözleşmeye yazar; IR’ı yeniden yazmaz.
- Sahte zamanı “yanlış bilim” sayıp implicit SINDy’yi `D(z)\dot x-N=0` yörünge formuna zorlamak. Unique-claim hedefi `D(z)` fonksiyonudur; `ẋ` değildir. API kokusu + ızgara occupancy sorunu vardır, formül hatası zorunlu değildir.
- `UnknownMechanism` tipi. Delik zaten `NeuralDestructionTerm`’dir ([src/MechanismCompiler.jl](src/MechanismCompiler.jl) ~113–119).
- HybridCompose/HybridResidual’ı “çift residual implementasyonu” diye birleştirmek. Production tek yerdedir; M1 bunları silerek “dedupe” yapmasın.

Kod envanteri (üç keşif taraması) plan kararlarını değiştirmiyor: `IdentifiabilityReport` fonksiyonel ID sayılmaz; `ExperimentSet` split değildir; graph-local kapıları analitik `D` üzerindedir; kök Manifest / persist benchmark yoktur.

## v1.0 bilimsel kimlik (sözleşme taslağı)

Bilinen:

- Kullanıcı grafı (`BiologicalNetwork`) doğru ve tam kabul edilir.
- Bilinen kinetik derlenir; nöral olmaz.
- Tam gözlem (unique-claim); maskeli UDE eğitimi kapalı kalır.

Bilinmeyen:

- Tam bir yıkım hızı \(D(z)\geq 0\), \(\dot u_i = P_i - D_i u_i\).
- Üretim deliği, topoloji keşfi, 2+ delik **iddia dışıdır**. `validate_network` delik saymaz; `assert_single_unknown_destruction` ayrı kapıdır. Bu ayrım korunsun.

Başarı katmanları (karışmayacak):

- **Q1 Predictive fit:** yörünge / hybrid residual (eğitim ve held-out).
- **Q2 Mechanism function:** \(\hat D(z)\) vs \(D_{\mathrm{true}}(z)\) (ızgara + yörünge occupancy + held-out \(r\)).
- **Q3 Scale / parameter practical ID:** `k_prod`↔`D` kolinerliği **beklenen sınırlama** olarak raporlanır; biyolojik sabit iddiası kapalı kalır.
- **Q4 Functional ID:** bağımsız eğitimlerde \(D_i(z)\) anlaşması vs yörünge anlaşması. Sertifika değil, tanı.
- **Q5 Symbolic support:** true-monomial recall; F1 iskelet; kanonik Hill-from-NN kapalı.
- **Q6 Constraints:** \(D\geq 0\), payda işareti, pozitif orthant yapısı (teorem değil).
- **Q7 Held-out generalization:** görülmeyen IC ve \(r\) bölgesi.

**Identifiability reframing (kritik sözleşme kararı):** Bugün unique-claim `unidentifiable_edge == true` **ister** ([src/UniqueClaim.jl](src/UniqueClaim.jl) ~420–423). Bu ölçek belirsizliğini gizlememek için dürüsttür ama “mekanizma belirlendi” ile ters yöndedir. v1.0: ölçek uyarısı **zorunlu rapor** olarak kalır; identifiability **başarısı** fonksiyonel tanı + held-out `D` olur. Eski kapı, yeni katman yeşil olana kadar migration süresince durur; sessizce silinmez.

**Eşik disiplini:** `data_residual = 0.30` körlemesine sıkılaştırılmaz (Kural 0.2). Önce held-out dağılımı ölçülür, gerekçe yazılır, sonra eşik değişirse breaking sayılır.

## En küçük tutarlı “research-grade” küme

v1.0’ı gerçekten araştırma kalitesine taşıyan minimum:

1. Yazılı v1.0 sözleşmesi (\(P-D\cdot u\), Q1–Q7, kapalı iddialar).
2. `run_recovery_suite` ayrışması + tipli recovery sonucu.
3. `ExperimentSet` train/holdout + held-out residual **ve** `D` hatası.
4. Pratik fonksiyonel identifiability tanısı (sertifika değil).
5. Çok tohum başarı oranı (release artifact; her PR’da N×40 dk değil).
6. Graph-local’ın **eğitilmiş** `D` üzerinde tekrarı.
7. Aynı görev, aynı split, aynı metrik baseline (önce iç: saf UDE + ExplicitSTLSQ).
8. Golden-path provenance + research lockfile + persist edilen metrik tablosu.
9. Honesty testlerini bilim testlerinden ayırma; string kilit büyütmeyi durdurma.

Bunun dışı (Bayes, OED, multi-hole, SBML, GPU, LLM, JOSS) research-grade v1.0 için gerekli değildir.

## Tip kararları — yalnızca gerçek sorunu çözüyorsa

- **`MechanismRecoveryResult` — A, evet.** `run_recovery_suite` `Dict{Symbol,Any}` + `build_protocol_result` NamedTuple karışımı Q1–Q7’yi ayıramaz. Tek bilimsel çıktı nesnesi olsun. [src/RecoveryAdmission.jl](src/RecoveryAdmission.jl) `UniqueClaimProtocolRow` yazıcı/parmak izi olarak kalsın.
- **`FunctionalIdentifiabilityDiagnostic` — A, evet.** `IdentifiabilityReport` parametre-only’dir. Yeni nesne: yörünge anlaşması, fonksiyon anlaşmazlığı, tohum listesi, domain tanımı. Adında “certificate” / “structural” olmasın.
- **`ExperimentSplit` — A, evet; `ExperimentSet`’i şişirmeyin.** `split_experiments(set; train, holdout)` + hangi IC’lerin nereye gittiğinin provenance’ı. `ExperimentSet` semantiğini bozmayın.
- **`RunMetadata` — A, mevcut tipi güçlendirin.** Yeni `Provenance` icat etmeyin. `config::Any`’yi typed/named yapın; golden path’e bağlayın; `data_hash` doldurun.
- **`ValidationReport` — B, ayrı tip değil.** `MechanismRecoveryResult` üzerinde Q-katman görünümü + `format_*`. İkinci paralel bilim nesnesi doğurmayın.
- **`MechanismHypothesis` — D / ertele.** Tek aday + `ImplicitCandidate` + `DiscoveryResult` var. Tam sıralama v1.0 dışı. Şimdi sarmak spekülatif zarafettir.
- **`UnknownMechanism` — D.** `NeuralDestructionTerm` yeterli.
- **`RecoveryOutcome` — B.** `DiscoveryRetcode` / `TrainingRetcode` kalsın. İnce bir birleşik sonuç enum’u (`ScaleNonidentifiable`, `FunctionalDisagreement`, `HoldoutMechanismFailure`, `OptimizationFailure`, …) yalnızca pipeline’ın ürettiği sınıflar için. [src/FailureModes.jl](src/FailureModes.jl) bu iş için yanlış yerdir (honesty string envanteri).

`UDEModel` `Any` sarmalayıcısı ([src/UDE.jl](src/UDE.jl) ~23–37) bilinçlidir (JET UnionAll kaçışı). v1.0’da tam parametrik rewrite **C**; önce golden-path `Any` sızıntısını ölçün, kırılgan rewrite yapmayın.

## A / B / C / D

**A — v1.0 zorunlu**

- `docs/src/design/v1_contract.md` (kod gerçeği: \(P-D\cdot u\), tek yıkım deliği).
- Recovery pipeline ayrışması + `MechanismRecoveryResult`.
- Held-out IC + held-out \(r\) üzerinde residual ve `D` hatası.
- Fonksiyonel identifiability tanısı.
- Ölçek uyarısının “mekanizma belirlendi” olmaktan çıkarılması (migration ile).
- Çok tohum başarı oranı (release/nightly artifact).
- Graph-local’ın eğitilmiş UDE `D` üzerinde gösterimi.
- Aynı-görev iç baseline.
- Golden-path `RunMetadata` + research lock + persist metrik.
- Q-katmanlı test/CI ayrımı; residual eşiğinin gerekçesi (sıkılaştırma kanıta bağlı).
- Kanonik Hill-from-NN kapalı kalır; F1 iskelet kalır.
- Fisher aralıklarını “credible interval” diye sunmama.

**B — v1.0 güçlü öneri**

- Yörünge-örnekli `D` keşfi (ızgaraya ek).
- Extras’ın held-out `D` üzerindeki fonksiyonel etkisi (cebirsel kanonikleştirme değil).
- `RecoveryOutcome` ince enum.
- Honesty → typed field assert migration (string kilit büyütmeyi durdur).
- Format’ı tüm `src/` + `test/`’e yayma; Aqua/JET’i extras’a alma; `standards`’ı bilinçli kırmızı olmaktan çıkarma.
- 0-delikte dummy NN kafasını kaldırma (`n_heads = max(n,1)`).
- Compat’i research lock’ta sıkı tutma; paket compat’i makul bırakma.
- Saf UDE (semboliksiz) vs hibrit residual karşılaştırması.

**C — iyi gelecek iş**

- Çok aday hipotez sıralama / Pareto.
- Cebirsel destek eşdeğerliği / kanonik Hill-from-NN (yeni major kapı).
- Gürültü + örnek yoğunluğu tam zarfı (script’ler var; v1.0’da nokta + bir orta gürültü yeter).
- `UDEModel` somut parametrikleştirme.
- Bayes UQ, ensemble kalibrasyonu.
- MM’de kanonik destek.
- 3/6-durum **eğitilmiş** UDE (analitik prior’ın ötesi).
- Harici DataDrivenSparse, resolve edilirse izole env.

**D — açıkça kapsam dışı**

- Tam hipotez ranking, OED, aktif öğrenme, sıralı mekanizma tamiri.
- Keyfi çok-delik, genel topoloji, genel CRN / SINDy / neural ODE çerçevesi.
- Geniş SBML MathML, GPU eğitim yığını, LLM.
- Yapısal identifiability (StructuralIdentifiability.jl).
- Yaş laboratuvar / lisanslı seri.
- JOSS. General kayıt **maintainer eylemi**, bilimsel v1.0 kabul kriteri değil.
- Vizyon denklemini implementasyona sessizce dayatma.

```mermaid
flowchart TD
    net[BiologicalNetwork]
    compile[compile_mechanism P minus D u]
    ude[UDEModel one NeuralDestructionTerm]
    split[ExperimentSplit train holdout]
    train[train_experiments on train]
    sampleD[sample D on trajectories and grids]
    disc[discover_unknown_rate]
    q1[Q1 residual train and holdout]
    q2[Q2 D function error]
    q3[Q3 scale collinearity warning]
    q4[Q4 functional diagnostic]
    q5[Q5 support recall extras]
    result[MechanismRecoveryResult]
    net --> compile --> ude --> split --> train
    train --> sampleD --> disc
    train --> q1
    sampleD --> q2
    train --> q3
    train --> q4
    disc --> q5
    q1 --> result
    q2 --> result
    q3 --> result
    q4 --> result
    q5 --> result
```

---

## Milestone 0 — v1.0 bilimsel sözleşmesi (P0, P6 başlangıç)

- **Hedef:** Kodun gerçekten çözdüğü problemi yazmak; vizyon denklemini miras almamak.
- **Bilimsel soru:** “Başarı” Q1–Q7’nin hangisidir? Hangisi kapalıdır?
- **Sorun:** `docs/src/design/` yok. README / [docs/src/stability.md](docs/src/stability.md) / vizyon üç farklı denklem/başarı dili kullanır.
- **Neden önemli:** Sonraki her kapı bu metne bağlanmazsa metrik kayması tekrarlar.
- **Dosyalar:** yeni [docs/src/design/v1_contract.md](docs/src/design/v1_contract.md); çapraz bağ [docs/src/unique-claim.md](docs/src/unique-claim.md), [docs/src/architecture.md](docs/src/architecture.md), [README.md](README.md), [docs/src/out-of-scope.md](docs/src/out-of-scope.md). Kod değişikliği yok veya yalnızca “sözleşme dosyası var / kilit cümleler duruyor” testi.
- **Mimari:** Belge katmanı. `RECOVERY_THRESHOLDS` ve export listesi değişmez.
- **Matematik:** \(\dot u_i=P_i-D_i u_i\); tek `NeuralDestructionTerm`; ölçek kolinerliği beklenen.
- **API:** Yok.
- **Testler:** Sözleşme dosyasının varlığı; yasaklı cümleler (`ẋ = f_known + D` ürün iddiası olarak, “canonical Hill from NN”, “structural identifiability certificate”).
- **Bilimsel doğrulama:** Mevcut kapıların Q-etiketleri (Q1 residual, Q3 edge, Q5 recall).
- **Benchmark:** Yok.
- **Dokümantasyon:** Sözleşmenin kendisi.
- **Kabul:** Bir araştırmacı “ne kurtarıldı?” sorusuna tek sayfada doğru cevap alır; vizyon §3 denklemi ürün iddiası değildir.
- **Riskler:** Honesty string testleri belge cümlelerine bağlı; cümle değişince kırmızı. Önce kilitleri envanterle, sonra yaz.
- **Rollback:** Dosyayı silmek.
- **Ertelenen:** Mechanism Repair döngüsü, OED, multi-hole.

---

## Milestone 1 — Recovery mimarisini ayrıştır (P1)

- **Hedef:** Simülasyon / eğitim / `D` örnekleme / keşif / değerlendirme / rapor ayrımı. `run_recovery_suite` ince bir düzenleyici olsun.
- **Bilimsel soru:** Yok (altyapı). Sonraki M2–M5’in test edilebilir olması.
- **Sorun:** [src/Recovery.jl](src/Recovery.jl) tek yığın; dummy RNG consume; `Dict{Symbol,Any}`; [src/RecoverySuiteSkip.jl](src/RecoverySuiteSkip.jl) kaynak gövdesine bağlı.
- **Neden önemli:** Held-out ve fonksiyonel ID bu yığına yama olarak girerse yeni sızıntı doğar.
- **Dosyalar:** [src/Recovery.jl](src/Recovery.jl), [src/TrainingReuse.jl](src/TrainingReuse.jl), [src/UniqueClaim.jl](src/UniqueClaim.jl), [src/RecoveryAdmission.jl](src/RecoveryAdmission.jl), [src/RecoverySuiteSkip.jl](src/RecoverySuiteSkip.jl), [src/DataGen.jl](src/DataGen.jl); yeni ince modül örn. `src/RecoveryPipeline.jl` (yalnızca orkestrasyon; honesty taşımayın).
- **Mimari:** `generate_recovery_experiments` → `fit_unknown_destruction` → `sample_destruction` → `discover_unknown_rate` → `evaluate_recovery` → `MechanismRecoveryResult`. Fixture ağları (`build_hill_recovery_network`) ayrı kalsın; örnek [examples/unknown_inhibition.jl](examples/unknown_inhibition.jl) public `ReactionSpec` yolunu korur (suite ile aynı Hill sınıfı, ayrı constructor). `run_recovery_suite` aynı section adlarını koruyan ince wrapper. HybridCompose/HybridResidual/IdentifiabilityProduct **taşınmaz, birleştirilmez**.
- **Matematik:** Yok; çağrı sırası korunur.
- **API:** Public export **büyümesin**. Yeni tipler `BioDynaX.foo` iç. `RECOVERY_THRESHOLDS`, `hybrid_data_residual`, `compose_hybrid_rhs` dursun.
- **Testler:** Mevcut hard job seed 103 bit-dışı ama kapı-eşit (aynı KPI’lar). Skip sayacı davranışını fonksiyon sayacı olarak yeniden bağla; `occursin("function …")` büyütme.
- **Bilimsel doğrulama:** Hill σ=0 seed 103 kapıları değişmeden yeşil.
- **Benchmark:** [benchmark/recovery_suite.jl](benchmark/recovery_suite.jl) aynı stdout semantiği.
- **Dokümantasyon:** [docs/src/architecture.md](docs/src/architecture.md) veri akışı.
- **Kabul:** Suite bölümleri bağımsız çağrılabilir; RNG consume hack’i belgelenir veya kaldırılır (tercih: her section kendi `MersenneTwister(seed)`).
- **Riskler:** Honesty string’leri `run_recovery_suite` gövdesine kilitli. **Aynı PR’da honesty silmeyin.** Önce bilimsel fonksiyonlar, sonra kilitleri typed assert’e taşıyın.
- **Rollback:** Wrapper’ı eski gövdeye yönlendirmek.
- **Ertelenen:** `UDEModel` rewrite, honesty silme, `MechanismHypothesis`.

---

## Milestone 2 — Held-out çoklu-deney doğrulama (P0, P2)

- **Hedef:** Residual ve `D` hatasını eğitim dağıtımından ayırmak.
- **Bilimsel soru:** Q7 — aday görülmeyen IC / \(r\) bölgesinde yaşar mı? Q1 ile Q2 ayrılır mı?
- **Sorun:** 9 IC’nin 8’i residual’da yok; `r` ızgarası eğitim extrema’sından şişer (`_regulator_grid`).
- **Neden önemli:** extras’lı `D` eğitim IC[1] residual 0.30’u kolay geçer; bu “mekanizma kurtarıldı” değildir.
- **Dosyalar:** [src/Experiments.jl](src/Experiments.jl), [src/Recovery.jl](src/Recovery.jl), [src/Identifiability.jl](src/Identifiability.jl), [src/DataGen.jl](src/DataGen.jl), [examples/unknown_inhibition.jl](examples/unknown_inhibition.jl), [test/test_recovery_hard.jl](test/test_recovery_hard.jl), [test/test_experiments.jl](test/test_experiments.jl).
- **Mimari:** `ExperimentSplit` (train / holdout IC indeksleri, isteğe holdout \(r\) aralığı). Değerlendirme: `hybrid_data_residual` train ve holdout; `rate_rel_rmse` / korelasyon train ızgarası, holdout ızgarası, yörünge occupancy. Identifiability hem ref IC hem holdout IC’de raporlanır (tek IC Fisher’ın zayıflığı belgelenir).
- **Matematik:** Holdout IC protokolü önceden kilitlenir (ör. 9 IC’den 2–3 holdout, seed’den bağımsız indeks — sızıntı yok). Holdout \(r\): eğitim `[lo,hi]` dışına taşan bant veya ayrı IC occupancy. `D` hatası scale-invariant olmalı (`normalize_destruction_samples` zaten var); hem ham hem normalize raporlanır.
- **API:** `split_experiments` iç. Public `ExperimentSet` kırılmasın.
- **Testler:** Split provenance; holdout’un train IC’leriyle kesişmediği; sentetik “ezberleyen” extras `D`’nin holdout `D` RMSE’de yakalanabileceği bir birim senaryo (fikstür-özel sihir değil).
- **Bilimsel doğrulama:** Seed 103 Hill: holdout residual ve holdout `D` RMSE raporlanır. İlk v1.0 kapısı: holdout `D` hatası sonlu + train’den sistematik olarak gizlenmez. 0.30’u holdout’a kopyalamayın.
- **Benchmark:** Holdout sütunları [docs/src/benchmarks.md](docs/src/benchmarks.md) ve persist tablo.
- **Dokümantasyon:** Sözleşme Q1 vs Q2 vs Q7; “IC[1] residual = protokol” cümlesi kalkar.
- **Kabul:** Unique-claim çıktısı `data_residual_train` ve `data_residual_holdout` + `d_rmse_holdout` taşır. Tek sayı 0.30 artık tek başarı anlatısı değildir.
- **Riskler:** Kötü split şansı. İndeksleri protokole kilitleyin, optimize etmeyin. Holdout çok küçükse gürültü; 9 IC’de 6/3 veya 7/2 makul.
- **Rollback:** Eski `ref_exp = first(...)` alanını `legacy_data_residual` olarak bir süre tutun.
- **Ertelenen:** Zaman penceresi holdout, girdi rejimi, kısmi gözlem UDE.

---

## Milestone 3 — Pratik fonksiyonel identifiability (P2)

- **Hedef:** “Yörünge uydu ⇒ `D` tek” yanılsamasını ölçmek.
- **Bilimsel soru:** Q4 — bağımsız \(\hat D_i\) bilimsel bölgede anlaşır mı, yoksa yalnızca \(x(t)\) mi anlaşır?
- **Sorun:** Tek fit, tek tohum, NN hariç Fisher. Unique-claim başarı olarak ölçek *belirsizliğini* ister.
- **Neden önemli:** Softplus kafa + `softplus(k_prod)` ölçek soğurur; extras `1`,`r` aynı yörüngeye oturabilir.
- **Dosyalar:** [src/Identifiability.jl](src/Identifiability.jl) (parametre tanısı kalsın), yeni ince `src/FunctionalIdentifiability.jl`, [src/UDE.jl](src/UDE.jl) / training girişleri, [src/UniqueClaim.jl](src/UniqueClaim.jl) ürün bloğu, testler.
- **Mimari:** `FunctionalIdentifiabilityDiagnostic`: tohumlar, ortak `z` domain (train occupancy ∪ holdout \(r\)), pairwise `D` RMSE / korelasyon (ölçek-normalize), pairwise yörünge RMSE, dürüst özet (`trajectory_agree_function_disagree` bayrağı). `m` küçük (3–5 restart) v1.0 için yeter.
- **Matematik:** Sertifika değil. Domain ve normalize kuralı belgede. `unidentifiable_edge` Q3 uyarısı olarak kalır; Q4 ayrı. `parameter_credible_intervals` yeniden adlandırılır: asymptotic Fisher interval; “credible” kalkar.
- **API:** Unexported. `production_destruction_tradeoff` imzası korunur.
- **Testler:** Sentetik iki `D` (ölçek katı vs şekil farkı) tanının ayırdığı. Hard job: tanı üretilir; “certificate” kelimesi yasak.
- **Bilimsel doğrulama:** Seed 103 + en az 2 ek tohumda Q1 yakın / Q4 uzak olabilir — bu geçerli negatif veya belirsizliktir, gizlenmez.
- **Benchmark:** Çok tohum `D` anlaşma tablosu.
- **Dokümantasyon:** [docs/src/identifiability-product.md](docs/src/identifiability-product.md) Q3 vs Q4.
- **Kabul:** Çıktıda Q3 ve Q4 ayrı. Hiçbir satır “functionally identifiable” demez; “practical functional diagnostic” der.
- **Riskler:** Küçük `m` ve dar domain sahte anlaşma. Domain’i occupancy’den üretin, `[0.05,2]` sabitine sapmayın. CI maliyeti: her PR’da tam `m` değil; birim test + nightly `m`.
- **Rollback:** Q3-only ürüne dönüş; tanı alanı opsiyonel.
- **Ertelenen:** Bayes fonksiyonel UQ, yapısal ID, global rank teoremleri.

---

## Milestone 4 — Kurtarma kanıtını sağlamlaştır (P3)

Üç bağlı ama ayrı iş; tek “başarı hikayesi” yazılmadan bitmez.

### 4a Yörünge bağlamında `D` örnekleme

- **Hedef:** Keşfi yalnızca sentetik 1D ızgara + sahte `t∈[0,1]` olmaktan çıkarmak.
- **Bilimsel soru:** `D` eğitimde gerçekten ziyaret edilen \((z,t)\)’de mi, yoksa düzgün \(r\) ızgarasında mı öğrenildi?
- **Sorun:** `_regulator_grid` + dummy times; `discover_equations` API’si `times` ister.
- **Mimari:** Birincil: IC yörüngelerinden `(z(t), D_nn(x(t)))`. Izgara ikincil / görselleştirme. Dummy time kalırsa adı `sample_index` veya keşif API’sine “times optional” — bilimsel iddia “dinamik SINDy” olmaz.
- **Kabul:** Unique-claim keşfi yörünge-örnekli `D` üzerinde çalışır; ızgara ablation olarak kalabilir.
- **Ertelenen:** `ImplicitCandidate` docstring’indeki `D(z)ẋ-N=0`’ı unique-claim’e zorlamak. Unique-claim `y=D_nn` fonksiyon regresyonudur; bunu dokümante edin ([docs/src/architecture.md](docs/src/architecture.md) satır 22 şu an yanıltıcı).

### 4b Graph-local eğitilmiş `D`

- **Hedef:** Prior sızıntısını kapatmak (analitik `D` ≠ yöntem).
- **Bilimsel soru:** Eğitilmiş UDE `D` üzerinde graph kütüphanesi yanlış ebeveyni düşürür mü?
- **Dosyalar:** [src/Recovery.jl](src/Recovery.jl) ablation, [src/GraphLocalLibrary.jl](src/GraphLocalLibrary.jl), [test/test_recovery.jl](test/test_recovery.jl).
- **Mimari:** 3-durum (veya 2-durum + distractor durum) unknown Hill → UDE eğit → `sample_*` → `scope=:graph` vs `:global` vs wrong-graph. Analitik ablation “library membership control” olarak etiketlenir, UDE iddiası olmaz.
- **Kabul:** En az bir eğitilmiş-`D` graph vs wrong-graph satırı CI’da (bütçe: 3-durum, kısaltılmış iterasyon kabul — kapı gevşetilmeden protokol küçültülürse açıkça `:ude_graph_prior` smoke vs protocol ayrılır).
- **Risk:** 40 dk iş. Ayrı section + kısa iterasyon smoke; tam protokol recovery/nightly.

### 4c Çok tohum başarı oranı

- **Hedef:** Tek şanslı optimizasyonun ürün olmaması.
- **Sorun:** [docs/src/out-of-scope.md](docs/src/out-of-scope.md) “N × 40 dk CI’ya ekleme” diyor — bu kısıt korunsun.
- **Mimari:** [benchmark/recovery_seeds.jl](benchmark/recovery_seeds.jl) persist JSON/CSV + `RunMetadata`. Release/nightly job. PR: seed 103 iskelet. v1.0 **iddiası** N tohumda başarı oranı (N≥5, önceden kilitli liste `(103,107,111,113,127)`). Başarısız tohum gizlenmez.
- **Kabul:** Medyan + başarı oranı yayın artifact’ında. “Typical” yalnızca en iyi tohum değildir.
- **Eşik:** Körlemesine N’den 1’e düşürmeyin; başarı tanımı M2/M3 katmanlarına bağlanır.

**4 ortak kabul:** Mechanistic recovery “recall + IC[1] residual + edge” olmaktan çıkar; en az holdout `D` + fonksiyonel tanı + tohum oranı + eğitilmiş graph-local kanıtı vardır.

**4 ortak ertelenen:** Tam gürültü ızgarası, sample-density merdiveni, 6-durum eğitilmiş UDE, kanonik Hill.

---

## Milestone 5 — Aynı-görev baseline (P3/P4)

- **Hedef:** Self-referential “BioDynaX vs BioDynaX graph/global” iddiasını kırmak.
- **Bilimsel soru:** Bu problemde (bilinen graf, tek `D(z)`, aynı örnekler) yöntem ne ekler?
- **Sorun:** DDS yok; skip ≠ win (dürüst). Saf UDE ve ExplicitSTLSQ aynı split’te yok.
- **Mimari:** Aynı `(R, D_nn veya D_true)`, aynı kütüphane, aynı train/holdout: (1) saf UDE — sembolik yok, Q1/Q2; (2) `ExplicitSTLSQ` on `D`; (3) mevcut `ImplicitSINDyPI`; (4) opsiyonel izole DDS. Metrikler: holdout `D` RMSE, recall, extras etkisi. Runtime ayrı sütun.
- **Dosyalar:** [benchmark/sindy_baseline.jl](benchmark/sindy_baseline.jl), [benchmark/probe_datadriven.jl](benchmark/probe_datadriven.jl), [test/test_baseline.jl](test/test_baseline.jl), [docs/src/benchmarks.md](docs/src/benchmarks.md).
- **Kabul:** En az (1)–(3) persist tablo. DDS `UNAVAILABLE` ise bilimsel kayıp yazılır, zafer yazılmaz. `continue-on-error` “geçti” sayılmaz.
- **Risk:** Zayıf baseline seçmek. ExplicitSTLSQ zaten in-tree; kasıtlı olarak zayıf bir rakip uydurmayın.
- **Ertelenen:** PySINDy, geniş yöntem yarışması, “üstünüz” iddiası kanıt yoksa.

---

## Milestone 6 — Yeniden üretilebilirlik ve provenance (P4)

- **Hedef:** Unique-claim satırını başka makinede pin’li ortamda yeniden üretmek.
- **Sorun:** Paket geleneği kök Manifest’i istemez; research sayıları pin’sizdir. `RunMetadata` kullanılmaz.
- **Mimari:** Kök `Project.toml` paket compat’i. Ayrı `research/Manifest.toml` (veya `benchmark/environments/recovery/`) gitignore istisnası. Artifact: seed, Julia, package, manifest hash, git SHA, ham KPI, split indeksleri. `save_result` / [src/ExperimentCheckpoint.jl](src/ExperimentCheckpoint.jl) golden path’e bağlanır.
- **API:** `RunMetadata.config` typed named tuple (training + discovery + split + solver).
- **Testler:** Fingerprint alanları dolu; smoke artifact protocol diye yaftalanamaz (`UniqueClaimFingerprint` kalsın).
- **Kabul:** `julia --project=research` (veya belgelenmiş komut) seed 103 tablosunu üretir; SHA + metrik dosyası vardır.
- **Risk:** Manifest drift. CompatHelper kökü günceller, research lock’u elle/CI’da yeniler.
- **Ertelenen:** GPU/thread pin’inin ötesi (`BLAS.set_num_threads(1)` dursun).

---

## Milestone 7 — Paket kalitesi ve kapı birleştirme (P5)

- **Hedef:** Bilim kapısı ile paket kapısını aynı “v1.0 yeşil” tanımında birleştirmek; her PR’da 40 dk bilim yığını koşturmamak.
- **Sorun:** Format kısmi; Aqua/JET geçici env; `standards` kırmızı olabilir; recovery tek OS.
- **Mimari (katmanlı CI, vizyon §53–54 ile uyumlu):**
  - **PR fast:** birim + IR + analitik keşif + split/tanı birim testleri + format **tüm src/test** + Aqua.
  - **PR recovery:** mevcut Ubuntu 1.10 hard job (iskelet).
  - **Nightly/release:** çok tohum + eğitilmiş graph-local + baseline tablosu.
  - **standards:** v1.0’da kırmızıya izin yok veya job “required” değilse v1.0 kesilmez — ikisini karıştırmayın. Tercih: JET/Aqua extras’ta, `quality` required, `standards` opt daraltılır.
- **UDEModel / dummy head:** 0-delik `n_heads=max(n,1)` kaldırma B; allocation gate’leri regresyon olarak kullanın ([benchmark/allocation_gate.jl](benchmark/allocation_gate.jl)).
- **Compat:** Research lock sıkı. Paket: `OrdinaryDiffEq="7"`, `Lux="1"` genişliğini ölçün; gerekçesiz daraltmayın, gerekçesiz genişletmeyin.
- **Kabul:** Format listesi kalkar (tüm Julia). Aqua extras’ta. Coverage `fail_ci_if_error` ayrı tartışılır; v1.0 bloğu değildir.
- **Risk:** Honesty + format tüm ağaçta kırılır. Önce format-only PR.
- **Ertelenen:** macOS recovery, Windows hard job, General kayıt.

**Honesty migration kuralı (bu milestone’un parçası, ayrı “temizlik şenliği” değil):** Yeni string kilit eklemeyin. Bilim değişince kilitleri **typed alan assert**’e taşıyın. `FailureModes.jl` / `*Honesty.jl` / `*Product.jl` / `ExperimentCheckpoint.jl` büyümesini durdurun. Toplu silme C — iddia gerilemesi riski yüksek. Gizli `Ref` sayaçlarını production semantiğinden ayırın; M1 skip davranışını fonksiyon sayacı olarak yeniden bağlasın.

---

## Milestone 8 — Dokümantasyon ve iddia hizası (P6)

- **Hedef:** Landing sayfaların M0–M5 ile aynı cümleyi söylemesi.
- **Sorun:** [docs/src/architecture.md](docs/src/architecture.md) unique-claim’i `D(z)ẋ-N=0` diye anlatır; gerçek `y=D_nn`. Vizyon 3080 satır charter’dır, implementasyon değil.
- **Dosyalar:** [README.md](README.md), [docs/src/index.md](docs/src/index.md), [docs/src/unique-claim.md](docs/src/unique-claim.md), [docs/src/architecture.md](docs/src/architecture.md), [docs/src/benchmarks.md](docs/src/benchmarks.md), [docs/src/stability.md](docs/src/stability.md), [docs/src/failure-modes.md](docs/src/failure-modes.md), [docs/src/howto.md](docs/src/howto.md), [CHANGELOG.md](CHANGELOG.md). Vizyon dosyasına “v1.0 implemented” yazmayın.
- **Kabul:** Q1–Q7 ayrı; held-out ve fonksiyonel tanı anlatılır; kanonik Hill kapalı; yaş veri yok; “not a general solver”.
- **Ertelenen:** Yeni tutorial ailesi (gürültü/belirsizlik örnekleri C), JOSS methods note.

---

## Milestone 9 — Performans bekçisi (P7, bloker değil)

- **Hedef:** M1–M5 regresyonunu yakalamak; yeni performans programı açmamak.
- **Sorun:** Allocation kapıları var; `UDEModel` `Any` hot-path riski bilinçli.
- **Kabul:** [benchmark/allocation_gate.jl](benchmark/allocation_gate.jl) ve mevcut sensealg sınırları (64/65) yeşil. Yeni allocation tavanı uydurmayın.
- **Ertelenen:** GPU, derleme stratejisi rewrite, tam type-stable UDE.

---

## Milestone 10 — Sürüm / topluluk (P8, bilimsel v1.0 sonrası)

- **Hedef:** Semver v1.0.0 + changelog + dondurulmuş public API. General / TagBot / JOSS **ayrı maintainer kapısı**.
- **Kabul (bilimsel v1.0):** A maddeleri yeşil; README hâlâ abartısız; export listesi bilinçli dondurulmuş ([docs/src/stability.md](docs/src/stability.md) freeze listesi gözden geçirilir, şişirilmez).
- **Ertelenen:** Registry, JOSS, community infrastructure.

---

## Bağımlılık ve teslim sırası

```mermaid
flowchart LR
    m0[M0 contract]
    m1[M1 pipeline]
    m2[M2 heldout]
    m3[M3 functional ID]
    m4[M4 robustness]
    m5[M5 baselines]
    m6[M6 provenance]
    m7[M7 package CI]
    m8[M8 docs]
    m9[M9 perf guard]
    m10[M10 tag]
    m0 --> m1 --> m2 --> m3 --> m4
    m2 --> m5
    m1 --> m6
    m4 --> m8
    m3 --> m8
    m6 --> m8
    m0 --> m7
    m7 --> m9
    m8 --> m10
    m4 --> m10
    m5 --> m10
```

M0 belgesiz M2 eşiği yazılmasın. M1’siz M2/M3 yığına yama olmasın. M4 gece işi M6 artifact olmadan “yayın kanıtı” sayılmasın.

## Bilinçli olarak yapılmayacaklar

Derleyiciyi \(f_{\mathrm{known}}+D\) yapmak; `validate_network`’e tek-delik kapısı koymak; residual 0.30’u kanıtsız 0.003 yapmak; F1’i 0.99’a boyamak; Hill-from-NN açmak; yaş veri uydurmak; her PR’da N×full UDE; honesty yığınını tek PR’da silmek; `MechanismHypothesis` public API; Bayes/OED/LLM/SBML/GPU.

## Literatür duruşu

v1.0 katkısı yeni bir keşif algoritması iddiası değildir. UDE + graph-local implicit STLSQ-PI + pratik Fisher uyarısı mevcut yöntemlerin **dar, dürüst, doğrulanmış birleşimidir**. Yenilik cümlesi yazılmayacak. Yakın bağlam: hybrid UDEs, SINDy-PI, practical identifiability / sloppiness, systems-biology’de gürültünün UDE’yi bozması. Her iddia cümlesi bir teste veya persist benchmark satırına bağlanır.
