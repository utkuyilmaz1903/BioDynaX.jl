Aşağıdaki denetim yalnızca mevcut kod, test ve belgelere dayanır. README cümleleri, izi koda ve teste düşmedikçe gerçek yetenek sayılmadı.

---

## 1. CURRENT SYSTEM MAP

Paket sürümü `0.9.2` (research preview). Kullanıcı yüzü `examples/unknown_inhibition.jl` ve `run_recovery_suite` / `test/run_recovery_hard.jl`. İkisi aynı protokol parmak izini paylaşır (`UNIQUE_CLAIM_PROTOCOL`: seed 103, 9 IC, 50 nokta, tspan `(0,8)`, Adam 100 / BFGS 50, bootstrap 8, discovery seed 3, gözlem gürültüsü 0). Örnek ağı `ReactionSpec` ile kurar; recovery job `build_hill_recovery_network` kullanır. Aynı Hill sınıfı, aynı IC tablosu.

Kanonik dinamik **vizyon belgesindeki** \(\dot x = f_{\mathrm{known}} + D(z)\) **değildir**. Derlenen biçim:

\[
\dot u_i = P_i(u,p,t) - D_i(u,p,t)\, u_i
\]

(`ude_rhs!` / `ude_system`). Bilinmeyen delik, toplanır artık değil; **çarpımsal yıkım hızı**dır.

| Aşama | Dosyalar | Tipler | Fonksiyonlar | Girdi → çıktı | Varsayımlar / kırılma | Test / benchmark |
|---|---|---|---|---|---|---|
| 1. Ağ | `src/Network.jl`, `src/Metadata.jl` | `BiologicalNetwork`, `NodeSpec`, `EdgeSpec`, `ReactionSpec`, `KineticFamily` | `BiologicalNetwork(...)`, `validate_network` | Kullanıcı grafı + reaksiyonlar → doğrulanmış ağ | İsim/sınır/stoikiometri/metadata. **Delik sayısı kontrol edilmez.** 0 veya ≥2 delik yasal | `test_network.jl`, `test_compiler.jl` |
| 2. Derleme | `src/MechanismCompiler.jl` | `CompiledMechanism`, `HillDestructionTerm`, `NeuralDestructionTerm`, … | `compile_mechanism`, `_reindex_neural_destruction` | Ağ → `P`/`D` IR. `known=false` → nöral kafa. Tekrarlayan kenar atlanır; kafalar `1:n` | `du = P − D·u`. Bilinmeyen yalnızca yıkım. En fazla 2 regülatör. `P` veya `D` boşsa hata | `test_compiler.jl`, `test_compiler_contract.jl` |
| 3. Bilinen/bilinmeyen ayrımı | Aynı IR + `src/RecoveryAdmission.jl`, `src/UniqueClaim.jl` | `NeuralDestructionTerm` | `neural_destruction_terms`, `count_unknown_destructions`, `assert_single_unknown_destruction`, `admit_recovery_suite_network` | Derlenmiş model → 0/1/2+ delik sayımı. Unique-claim **sonra** `n==1` ister | Derleyici açık kalır. Unique-claim kapısı ayrı | Fast suite + admission testleri. Recovery CI yalnızca 1 delik eğitir |
| 4. UDE | `src/UDE.jl`, `src/ParameterSchema.jl` | `UDEModel` (`nn`/`st`/`compiled` = `Any`), `MultiHeadNetwork` | `build_ude_model`, `build_ude_nn`, `pack_parameters` | RNG + ağ → Lux MLP (softplus çıkış) + `ComponentVector(phys, nn)`. **0 delikte bile `n_heads = max(n,1)`** | Fiziksel parametreler `softplus`; durumlar `max(0,x)` | `test_ude.jl`, `test_parameter_schema_pack.jl` |
| 5. ODEProblem | `src/SciMLInterface.jl`, `src/SciMLSolveSurface.jl` | `CompiledOOPRhs`, `SolverConfig` | `ODEProblem(model,u0,tspan,p)`, `build_ude_function`, `ude_system`, `ude_rhs!` | Model + `p` → SciML problem | Tsit5 varsayılan. Nöral delikte `InterpolatingAdjoint` + ZygoteVJP kilitli | `test_sciml_interface.jl`, `test_sciml_solve_surface.jl` |
| 6. Çözüm / veri | `src/DataGen.jl`, `src/DataGenContract.jl`, `src/Training.jl` | `GroundTruthModel`, `Experiment`, `ExperimentSet` | `compile_ground_truth_model`, `generate_from_compiled_model`, `generate_experiment_set`, `unique_claim_experiment_set`, `predict_ude` | Truth ağı + IC’ler → yörüngeler. Unique-claim σ=0 | Aynı derlenmiş ağ her IC’de. `:hill_p53_fixture` ayrı, eski p53 ODE | `test_datagen_contract.jl`, `test_compiled_path.jl` |
| 7. Eğitim | `src/Training.jl`, `src/TrainingReuse.jl`, `src/OptimizationInterface.jl` | `TrainingConfig`, `TrainingResult`, `TrainingSolveSession` | `train_ude`, `train_experiments`, `train_experiments_with_warmup`, `loss_mse` | Gözlem MSE (+ isteğe AL ceza). Adam sonra BFGS (BFGS tüm set) | İlk IC ısınması + horizon. `frozen_phys` gradyanı sıfırlar; `k_prod`↔`D` kolinerliğini kaldırmaz. Kayıp yalnızca yörünge uydurması | `test_training.jl`, `test_training_reuse.jl`. Unique-claim eğitimi **fast suite’te yok** |
| 8. Nöral yıkım örnekleme | `src/Recovery.jl` | — | `sample_unknown_destruction`, `sample_unknown_destruction_grid`, `_regulator_grid` | Eğitilmiş `D` derlenmiş katkı olarak (ham Lux değil). Grid: diğer durumlar `fill_value=0.3` | Grid, eğitim yörüngesi değil. `sample_learned_function` hâlâ var; unique-claim onu kullanmaz | `test_recovery.jl` (derlenmiş `D` eşleşmesi) |
| 9. Sembolik keşif | `src/Discovery.jl`, `src/BasisFactory.jl`, `src/DiscoveryWorkspace.jl` | `DiscoveryConfig`, `ImplicitSINDyPI`, `ImplicitCandidate`, `LocalBasisSpec` | `discover_unknown_rate` → `discover_equations`; `_stlsq` / implicit tasarım; `prune_nested_implicit` | Unique-claim: `R`×sahte `times∈[0,1]` + `derivatives = D_nn`. Kimlik `N/D ≈ D_nn`, **`D(z)\dot x − N = 0` yörünge anlamında değil** | ≥20 örnek. Payda tabanı. Occam BIC’i fit setinde; hold-out RSS güvenlik | Analitik Hill: `test_recovery.jl`. UDE→keşif: yalnızca hard job |
| 10. Identifiability | `src/Identifiability.jl`, `src/IdentifiabilityProduct.jl` | `IdentifiabilityReport` | `trajectory_jacobian` (FD), `assess_identifiability`, `production_destruction_tradeoff`, `unidentifiable_edge_from_fisher` | Fiziksel parametre Fisher’ı + `k_prod` Jacobian ile `D` ölçek pertürbasyonu kosinüsü | NN dışlanır. Eşik: `cond≥1e6` **veya** `cos≥0.95`. **Yapısal değil.** Unique-claim **`unidentifiable_edge == true` ister** | Fast: bayrak üretilir. Hard: `true` ve `cos≥0.95` |
| 11. Recovery değerlendirme | `src/Recovery.jl` | `RECOVERY_THRESHOLDS` | `support_f1`, `active_support`, `hill_rate_support`, `_evaluate_unknown_rate_recovery`, `hybrid_data_residual` | NN–truth korelasyon/RMSE; monomial recall/F1; hybrid RMSE (ilk IC); extras | Hill truth `{r^n}` pay ve paydada. F1 etiketli `(:n/:d, key)`. Residual eşiği **0.30** | Hard CI: Hill recall/residual/edge. MM: NN RMSE+residual, recall kapısı yok |
| 12. Rapor | `src/Recovery.jl`, `src/UniqueClaim.jl` | `UniqueClaimFingerprint`, `UniqueClaimProtocolRow` | `format_protocol_result`, `build_protocol_result`, `locked_ude_kpis` | IDENTIFIABILITY → FIT → DISCOVERY → REPRODUCTION. `canonical_hill_from_nn` **her zaman false** | Örnek recall/F1 **hesaplamaz**; yazdırır: “CI gate… not scored here” | Çoğu test yazıcı/alan kilidi |

Bağımlılıklar (doğrudan): Lux, Zygote, SciMLSensitivity, OrdinaryDiffEq, Optimization/Optimisers, ComponentArrays, Graphs, StaticArrays, ChainRulesCore. Zayıf: CUDA, Catalyst, DataDrivenSparse, ModelingToolkit, Plots, SBML, SBMLToolkit. `Manifest.toml` yok (`.gitignore`). `compat` geniş (`OrdinaryDiffEq = "7"`, `Lux = "1"`).

---

## 2. CURRENT SCIENTIFIC CLAIM

Kodun (`RECOVERY_THRESHOLDS`, `unique_claim_kpis_hold`, `build_protocol_result`, `stability.md`) savunduğu ürün, şu üç kapının **birlikte** yeşil olmasıdır:

1. **`unidentifiable_edge == true`** — pratik Fisher koşul sayısı veya `k_prod` ile `D` ölçek Jacobian kosinüsü. Bu bir “mekanizma belirlendi” iddiası değil; **ölçeğin belirlenemediğinin** kapısıdır. `coefficients_are_biological_constants` bağımsız ölçüm değil: `!unidentifiable_edge`.
2. **Hybrid residual ≤ 0.30** — `compose_hybrid_rhs` yörüngesinin **gözleme** RMSE’si (UDE \(\dot x\)’ine değil). Tipik README değeri ~0.003; kapı iki mertebe gevşek.
3. **True-monomial recall ≥ 0.99** — keşfedilen implicit destekte Hill truth monomialinin (`r^n` pay **ve** payda) hatırlanması. Katsayı veya kanonik form değil.

Bilinçli olarak **kapalı** tutulanlar: kanonik Hill’in eğitilmiş NN’den çıkarılması (`canonical_hill_from_nn` sabit `false`); combined F1’in UDE iddiası olması (taban 0.50); katsayıların biyolojik sabit olması; genel CRN / bilinmeyen topoloji / tek gürültülü CSV / eksik durumda UDE eğitimi / StructuralIdentifiability.jl.

**Terimlerin bu depodaki anlamı (eşadlı formalizmlerle karıştırılmamalı):**

| Terim | Bu depodaki anlam |
|---|---|
| **known graph** | Kullanıcının verdiği `BiologicalNetwork`. Çıkarılmaz. Unique-claim, grafın doğru ve tam olduğunu varsayar. |
| **one unknown destruction** | Tam bir `NeuralDestructionTerm`: `D(z)` hızı, `du_target -= D * u_target`. Derleyici 0/2+ deliğe izin verir; unique-claim yolu `n==1` ister. |
| **practical identifiability** | Bir fit noktasında, fiziksel parametrelerin sonlu fark yörünge Jacobian’ından Gauss–Newton Fisher. Yerel, asemptotik, NN’siz. |
| **functional identifiability** | **Yok.** Çoklu `D_1…D_m` karşılaştırması, fonksiyonel anlaşmazlık metriği veya vizyon §11 nesnesi implemente değil. |
| **mechanistic recovery** | Hill sınıfında: monomial recall + data residual + “ölçek belirlenemez” uyarısı. Katsayı kurtarma, cebirsel eşdeğerlik veya tek fonksiyonun benzersizliği değil. |
| **symbolic discovery** | Aynı monomial kütüphanede implicit STLSQ-PI + iç içe Occam. Unique-claim’te girdi, eğitilmiş `D(r)` örnekleri; sahte zaman ekseni. |
| **graph-local discovery** | `local_basis(scope=:graph)` = hedef + `inneighbors`. `:global` ablation. Kilit kanıt: kütüphane üyeliği / `local_has_true_parent_gate`, Occam sonrası F1 farkı değil. |
| **hybrid residual** | Sembolik `rate_fn` nöral `D`’nin yerini alır; RMSE gözleme karşı. |
| **true-monomial recall** | Etiketli destek kesişimi / truth. Hill n=2: truth 2 anahtar `(:n,r²)`, `(:d,r²)`. |
| **combined support F1** | Aynı etiketli kümede F1. Extra `1` ve `r` precision’ı düşürür (~0.5–0.67). UDE kapısı 0.50; analitik 0.99. |
| **canonical Hill recovery** | `vmax r^n / (K^n + r^n)` ve yalnızca o monomialler. **Kapalı.** Analitik `D` üzerinde açık; eğitilmiş NN’de extras kalır. |

---

## 3. ACTUAL IMPLEMENTED CAPABILITIES

Çalışan, koda bağlı yüzey:

- Bilinen kinetik IR: mass-action, lineer yıkım, Hill, MM (satürasyon), kompetitif, custom evaluator. `du = P − D·u`.
- Bilinmeyen yıkımı Lux MLP (softplus) olarak derleme; 1–2 regülatör; çoklu kafa ve `1:n` yeniden indeks.
- SciML `ODEProblem` / `solve` / `predict_ude`; nöral delikte adjoint kilidi.
- Çoklu IC sentetik veri: bir kez derle, her IC’yi aynı modelden üret.
- Yörünge MSE ile Adam+BFGS; isteğe horizon, `frozen_phys`, maskeli MSE (eğitim iddiası unique-claim’de yok).
- Eğitilmiş `D`’yi derlenmiş yoldan örnekleme; regulator-grid keşif.
- Graph-local implicit rasyonel keşif + Occam + payda ızgarası + `DiscoveryRetcode`.
- Analitik Hill’de (gürültüsüz / %0.5) combined F1 ≥ 0.99.
- Hill-sınıfı unique-claim (seed 103, 9 IC, σ=0): NN–Hill korelasyon/RMSE, recall ≥ 0.99, F1 ≥ 0.50 ve **< 0.99**, residual ≤ 0.30, `unidentifiable_edge`, extras dolu.
- σ=0.02 Hill hard job: aynı residual/recall kapıları; F1 tabanı 0.50.
- MM unknown: NN RMSE + residual; **Hill recall 0.99 uygulanmaz** (yorumda ~0.5 recall / ~0.33 F1).
- Bilinen-parametre recovery (gevşek RMSE): lineer / MM / Hill / kompetitif, tek IC, σ=0.
- Graph-prior: 3- ve 6-durum analitik `D` + yanlış-graf negatif (true parent kütüphanede yok).
- Identifiability müdahaleleri: dondurma / `k_prod` pertürbasyonu / `D` normalizasyonu tradeoff’u **kırmıyor** (kilit bulgu).
- Kısmi gözlem: **analitik** altörneklenmiş `D` → hybrid residual; maskeli lineer eğitim ayrı ve iddia dışı.
- Dürüstlük kilitleri: export listesi, eşik sayıları, smoke≠protocol, Hill-from-NN kapalı, lisanslı seri yok.

---

## 4. CLAIMS NOT CURRENTLY SUPPORTED

Sınıflandırma: **A** doğrudan doğrulanmış · **B** zayıf doğrulanmış · **C** kısmi · **D** yazılı ama yok · **E** kanıtsız.

| İddia | Sınıf | Gerekçe |
|---|---|---|
| Kullanıcı grafı derlenir; delik `NeuralDestructionTerm` olur | **A** | IR + testler |
| Unique-claim tam bir delik ister; 0/2 derlemede açıktır | **A** | Admission + validate ayrımı |
| Analitik Hill’de sparse rasyonel destek (Occam, aynı kütüphane) | **A** | Fast suite, F1 0.99 |
| Seed 103 Hill UDE: recall + residual + `unidentifiable_edge` | **A** (tek tohum, σ=0) | `test_recovery_hard.jl` CI |
| Combined F1 UDE’de kanonik Hill değildir | **A** (negatif) | F1 < 0.99 ve extras zorunlu |
| `unidentifiable_edge` pratik Fisher/kosinüs, yapısal değil | **A** hesap; **B** yorum | Tek yörünge, yerel, eşik or-kapısı |
| Hybrid residual “mekanizmayı kurtardı” | **B** | İlk IC; eşik 0.30; extras’lı `D` de geçebilir |
| Tipik residual ≈ 0.003 | **B** | README; kapı 0.30; örneğin recall’ı yok |
| σ=0.02 Hill UDE | **B** | Hard job var; gürültü ızgarası CI değil |
| Graph-local, yanlış ebeveyni kütüphaneden düşürür | **A** | Analitik `D`, eğitilmiş NN değil |
| Graph-local Occam sonrası F1’de global’i yener | **E** | Kod ve yorum: F1 eşitlenebilir; kilit kütüphane üyeliği |
| Fonksiyonel identifiability | **D / E** | Vizyon §11; kodda yok |
| Çoklu tohumda UDE recovery | **C** | `recovery_seeds.jl --ude` rapor; kırmızı kapı 103/104 |
| Gürültü / örnek yoğunluğu zarfı | **C** | Script’ler var; CI tek nokta |
| Held-out IC / held-out `D` bölgesi | **D** | Residual eğitim IC[1]; keşif hold-out Occam içi |
| SINDy / DataDrivenSparse üstünlüğü | **D / E** | Probe `continue-on-error`; “skip ≠ win” |
| MM’den kanonik MM destek | **E** | Bilinçli kapatılmış |
| Kanonik Hill from NN | **E** | Sabit `false` |
| Katsayı = biyolojik sabit | **E** | Ürün aksini kapılar |
| Eksik durumda UDE eğitimi | **E** | `ude_mask_train_claimed = false` |
| Yaş laboratuvar / lisanslı seri | **E** | Yokluk kilitli |
| Genel CRN, bilinmeyen topoloji, 2–20 durum ürünü | **D** | Kapsam cümlesi; ölçülen 2/3/6 |
| SBML MathML → Hill/MM, GPU eğitim, MTK ürünü | **D** | Extension, experimental |
| Yapısal identifiability | **E** | Açıkça hariç |
| `ẋ = f_known + D` (vizyon) | **E** | Gerçek biçim `P − D·u` |

---

## 5. SCIENTIFIC RISKS

**Yörünge uydurması ≠ mekanizma.** Eğitim kaybı yalnızca MSE. Unique-claim residual’ı da yörünge RMSE. Extra monomial’li (`1`, `r`) bir `D` residual kapısını (0.30) kolayca geçer. Recall, destekte `r^n` var mı diye bakar; cebirsel sadeleşme veya fonksiyonel teklik yoktur.

**Parametre / fonksiyon karışması.** Ürünün birincil “identifiability” bloğu tam da `k_prod` ile `D` ölçeğinin koliner olmasıdır ve **başarı için `true` gerekir**. Freeze, normalizasyon ve pertürbasyon bunu kırmaz — bu dürüst bir bulgu, ama “mekanizma belirlendi” ile ters yöndedir. `coefficients_are_biological_constants` ayrı analiz değil.

**NN ölçek soğurması.** Softplus kafa + `softplus` `k_prod` + `normalize_destruction_samples` aynı kütüphanede extras bırakır. Ölçek belirsizliği hem identifiability hem keşif F1’ine sızar.

**Yerel vs küresel identifiability.** Fisher bir fit’te, bir IC’de, sonlu fark. `unidentifiable_edge` or-kapısı (`cond` veya kosinüs). Çoklu rejim / global rank yok. `identifiability` suite bölümü **bilinen** Hill ağında çalışır (nöral kosinüs NaN).

**Benchmark sızıntısı.** Truth destek `hill_rate_support(2)` sabit. Aynı monomial sözlüğü. Unique-claim grid, eğitim IC’lerinin regulator aralığından üretilir. Keşif zamanı sahte. Ablation/graph-prior **analitik** `D` (+ %0.5 gürültü), eğitilmiş UDE `D` değil. İlk IC residual = eğitim dağılımı.

**Tohum duyarlılığı.** Kırmızı kapı 103 (Hill) / 104 (ablation). `recovery_seeds.jl --ude` CI’da yok.

**Fikstür mantığı.** 9 sabit IC, sabit truth `(k_prod=0.9, vmax=1.8, K=0.55, …)`, `fill_value=0.3`, Hill n=2, 2 durum. `run_recovery_suite` RNG’yi “tüketmek” için dummy `build_ude_model` çağırır.

**Graph-localite.** Prior doğru ebeveyni içeriyorsa kütüphane onu tutar; yanlış grafta true parent yoktur — bu topoloji testi, keşif gücünün kanıtı değil. `screen_variables` yüksek indegree’de korelasyon keser (unique-claim 1-regülatörde önemsiz).

**Sembolik eşdeğerlik.** Destek anahtarı `((vars),(powers))` tam eşleşme. `r^2/(K^2+r^2)` ile `(c r^2)/(c K^2 + c r^2)` farklı katsayı, aynı destek. `1+r` extras’ı “farklı fonksiyon” sayılmaz; F1’de fp’dir.

**Gizli sabitler.** Eşikler, protokol, IC tablosu, `rel_step`, `collinearity_threshold=0.95`, `condition_threshold=1e6`, `data_residual=0.30`, sahte `times`, `domain_samples`, `chunk_size`. `canonical_hill_from_nn` hiç hesaplanmaz.

**Sayısal.** Implicit tasarım `N − D_den * y`; payda tabanı; `eps` Hill/MM’de; `max(0,x)`; Zygote + mutating cache ayrımı; `pinv` Fisher kovaryansı.

**Biyolojik varsayımlar.** Her bilinmeyen yıkımdır; üretim deliği yok. `D≥0` mimari. Pozitif orthant `P−D·u` + `max(0,x)` ile “korunur” — bu teorem değil, yapı. LATENT durumlar ODE’dedir ama unique-claim tam gözlem kullanır.

**Yanıltıcı metrikler.** Combined F1 “iskelet tabanı” olarak durur ama protokol nesnesinde basılır. README ~0.57 / ~0.003 kapılardan sıkı; kapılar gevşek. Örnek recall’ı skorlamaz. `practical_not_structural = true` her satırda sabit.

**Zayıf negatifler.** σ=0.05 analitik Hill’in kırılması (`test_invariants.jl`) iyi bir negatif. MM recall kapısız. Yanlış-graf yalnızca üyelik. 2+ delik “eğitme, hata ver” — bilimsel karşı-örnek değil.

**Held-out yok.** 9 IC’nin 8’i residual’da kullanılmaz. Yeni IC / yeni `r` aralığı / yeni `k_prod` rejimi kapısı yok.

**Zayıf baseline.** İç ablation (graph vs global kütüphane). DataDrivenSparse çözülmezse skip. Saf UDE veya harici SINDy-PI karşılaştırması CI’da yok.

---

## 6. SOFTWARE ENGINEERING RISKS

- **İki API.** Export dondurulmuş; bilimsel ürün `BioDynaX.foo` ile (dondurulmamış, semver’siz). `UDEModel` alanları `Any` — JET/kararlılık baskısı.
- **Dürüstlük katmanı şişmesi.** `UniqueClaim.jl`, `IdentifiabilityProduct.jl`, `HybridResidual.jl`, `HybridCompose.jl`, `GraphLocalLibrary.jl`, `ClaimMetricHonesty.jl`, `ClaimScopeHonesty.jl`, `TrainingReuse.jl`, `SciMLSolveSurface.jl`, … kaynak/döküman **string** kilitleri. Bilimsel çekirdek (`Discovery.jl`, `MechanismCompiler.jl`, `Training.jl`) ile karışıyor; `run_recovery_suite` simülasyon+eğitim+keşif+rapor tek fonksiyon (~600 satır).
- **Çift yollar.** Reaksiyon ve kenar derlemesi; `sample_unknown_destruction` vs `sample_learned_function`; `discover_equations(ẋ)` vs `discover_unknown_rate(D)`; 0-delikte dummy NN kafası; `DEFAULT_EXAMPLE_NETWORK` (p53/Mdm2) varsayılan.
- **Gizli durum.** `DEFAULT_EXAMPLE_NETWORK`; `_note_compile_network` / `_note_train_unknown_edge` sayaçları; Lux setup yükleme zamanı sabitleri.
- **Bağımlılık / compat.** Manifest yok. `OrdinaryDiffEq 7`, `Lux 1` geniş. Aqua/JET `Project.toml` extras’ta değil; geçici CI ortamı. `external-baseline` `continue-on-error`. Coverage `fail_ci_if_error: false`.
- **Tip / allocation.** `UDEModel` sarmalayıcı bilinçli olarak parametresiz; hot path `Any` üzerinden. Allocation kapıları var ama format CI **dosya listesi** — tüm `src/` değil.
- **Kırılgan testler.** Kaynak `occursin("function …")`, yasaklı cümleler, export kümesi. Refactor bilim değiştirmeden kırmızı yapar; gevşek RMSE/residual bilim bozulunca yeşil kalabilir.
- **CI.** Fast test: Ubuntu/Windows 1.10+1, tek macOS×Julia 1. Recovery: **yalnızca Ubuntu × Julia 1.10**. Standards ayrı ve “kırmızı olabilir”. Format kısmi.
- **Dokümantasyon sapması.** Landing sayfalar iddiada görece dürüst. `docs/src/`’nin çoğu kilit cümle tekrarı. `docs/research/BIO-DYNAX-VISION.md` (3080 satır) uygulanmamış charter; mevcut mimariyle (özellikle `ẋ = f_known+D`) uyumsuz.
- **Benchmark tekrarlanabilirliği.** `benchmark/recovery_suite.jl` yazdırır, artifact commit etmez. Ortam pin’i yok.

---

## 7. TESTING AND VALIDATION AUDIT

**Katmanlar (fiili):** (1) birim/IR, (2) sayısal parite (solve/compose/workspace), (3) analitik keşif + graph-prior, (4) unique-claim UDE (ayrı job), (5) paket QA (Aqua/JET/format), (6) iddia/döküman string kilitleri.

**Bilimsel doğrulama (iddia koruyan):**

| Test | Koruduğu iddia | Tür |
|---|---|---|
| `test_recovery_hard` Hill σ=0 seed 103 | Recall + residual + edge + F1∈[0.50,0.99) + extras | **Bilimsel (tek tohum)** |
| Aynı, σ=0.02 seed 113 | Aynı kapılar gürültüde | **Bilimsel, zayıf zarf** |
| Hard MM | NN RMSE + residual; Hill recall **yok** | **Bilimsel negatif / dar iddia** |
| Analitik Hill + Occam %0.5 | Aynı kütüphanede kanonik destek mümkün | **Bilimsel, UDE değil** |
| σ=0.05 analitik kırılır | Keşif gürültüye duyarlı | **Dürüst negatif** |
| Ablation / 3- / 6-state / wrong-graph | Prior üyelik; yanlış graf true parent’ı düşürür | **Bilimsel, analitik `D`** |
| `ident_interventions` | Freeze/ölçek tradeoff’u kırmaz | **Bilimsel (identifiability)** |
| `partial_obs` | Altörnek analitik `D` → residual; mask-train iddia dışı | **Kısmi bilimsel + kilit** |
| Bilinen lineer/MM/Hill/kompetitif RMSE | Derleyici+eğitim parametre kurtarır | **Zayıf bilimsel** (eşikler 0.25–0.55) |
| Hybrid compose/residual parite | Residual tanımı tutarlı | **Sayısal; recovery değil** |
| Pozitiflik `P−D·u` | Sınırda işaret | **Nümerik invariant** |

**Uygulama / sözleşme testleri (bilim değil):** export listesi, eşik kopyası, protokol alan sırası, extras `NA`/`(none)`, smoke≠9 IC, `validate_network` kaynak taraması, `canonical_hill_from_nn === false`, format/JET/Aqua, workspace `resize_count`, sensealg 64/65, compile-once sayaçları, Elowitz “experimental_csv == false”, DataDrivenSparse yoksa mesaj.

**Boşluk:** Fast suite unique-claim UDE eğitmez (`test_example_smoke` 1 IC / 2 Adam). Recall örneğinde skorlanmaz. Çoklu tohum, held-out IC, fonksiyonel `D` anlaşmazlığı, harici baseline yok. Graph-prior eğitilmiş NN `D` kullanmaz.

---

## 8. REPRODUCIBILITY AUDIT

| Öğe | Durum |
|---|---|
| RNG | `MersenneTwister(seed)` eğitim ve keşifte; keşif permütasyonu `seed ⊻ golden` |
| Protokol tohumları | 103 eğitim, 3 keşif, 104 ablation; hard gürültü 113 |
| Ortam | `Project.toml` + geniş compat; **Manifest yok** (paket geleneği, unique-claim sayılarını pin’lemez) |
| Sürüm | `PACKAGE_VERSION = v"0.9.2"`; `RunMetadata` (seed, Julia, hash) discovery’de var, golden path persist etmez |
| Checkpoint | Julia `serialize`; şema `1.0.0` |
| CI | Recovery 1.10 Linux; fast matrix 1.10/1.x. Aynı sayı Windows/macOS/Julia 1’de garanti değil |
| Benchmark | Script stdout; commit SHA / ortam / tablo artifact yok |
| CSV | `examples/data/unknown_inhibition.csv` sentetik; smoke üzerine yazmaz |
| GPU/thread | `BLAS.set_num_threads(1)` testlerde |

Tek makinede seed 103 tekrarı **mümkün**; yayın-kalitesi pin’li ortam **yok**.

---

## 9. ARCHITECTURAL STRENGTHS

Bunlar vizyon için gerçekten sağlam çekirdekler:

1. **Derlenmiş `P−D·u` IR** — bilinen kinetik nöral hale gelmez; delik tek terim.
2. **SciML yüzey** — aynı `UDEModel` generate / train / residual’da.
3. **Delik politikasının derleyiciden ayrılması** — 0/2 delik yasal; unique-claim kapısı ayrı.
4. **Keşif `DiscoveryRetcode`** — sessiz başarı yok.
5. **Graph-local kütüphane vs `:global` ablation** — prior açık.
6. **İddia daraltması** — F1-from-NN ve wet-lab kapatılmış; bu bilimsel hijyen.
7. **Pratik `k_prod`↔`D` tanısı** — ölçek belirsizliğini gizlemiyor.
8. **Çoklu IC + tek derleme + warmup** — mühendislik olarak doğru yön.
9. **Eşik gevşetmeyi breaking sayma** — metrikle oyuna karşı süreç (eşiklerin kendisi gevşek olsa da).

---

## 10. v1.0 BLOCKERS

`docs/src/stability.md` yeşil recovery’nin yeterli olmadığını söyler. Mevcut kanıta göre v1.0’ı bloke edenler:

1. **Kanonik / fonksiyonel mekanizma kurtarma yok** — extras kalır; Hill-from-NN kapalı; F1 0.50.
2. **Tek tohum, tek fikstür, σ∈{0, 0.02}** — vizyon §12–14 karşılanmıyor.
3. **Held-out IC / bölge yok** — vizyon Q7.
4. **Fonksiyonel identifiability yok** — vizyon §11, §36.
5. **Baseline yok** — DataDrivenSparse çözülmüyor; karşılaştırma self-referential.
6. **Residual kapısı bilimsel olarak zayıf (0.30)** + residual yalnızca IC[1].
7. **Yaş veri yok** — dürüst, ama v1.0 “araç” iddiasını keser.
8. **Kayıt / tag yok** — General’de değil; Manifest yok.
9. **`UDEModel` / public-vs-`BioDynaX.foo` istikrarsızlığı.**
10. **Vizyon denklemi ile implementasyon denklemi uyumsuz** — v1.0 sözleşmesi yazılmadan (`docs/src/design/v1_contract.md` yok) “ne kurtarıldı?” belirsiz.
11. **Standards job bilinçli olarak kırmızı olabilir** — paket olgunluğu eşiği ayrı.
12. **`run_recovery_suite` tek yığın** — v1.0 katmanlı mimari değil.

---

## 11. HIGH-VALUE IMPROVEMENT AREAS

Önce bilim, sonra mühendislik (uygulama yok; yalnızca boşluk):

1. Held-out IC residual ve held-out `r` ızgarasında `D` hatası — yörünge vs mekanizma ayrımı.
2. Fonksiyonel tanılama: aynı veri, bağımsız tohumlarda `D_i(z)` anlaşması; yörünge yakın / fonksiyon uzak.
3. `data_residual=0.30` yerine ölçülmüş, gerekçeli eşik; F1’i iddia dışı bırakmaya devam.
4. Unique-claim’i N tohumda **kapı** yapmak (rapor değil); başarı oranı.
5. Graph-prior’u **eğitilmiş** `D` üzerinde tekrarlamak (analitik `D` sızıntısını kapatmak).
6. Cebirsel/fonksiyonel destek eşdeğerliği (yalnızca monomial anahtar değil).
7. Uygun baseline: aynı `D` örnekleri, aynı kütüphane, aynı split.
8. `Manifest.toml` veya lockfile’lı reproduction artifact (seed, SHA, metrikler).
9. `run_recovery_suite` ayrıştırma; honesty string testlerini bilim testlerinden ayırma.
10. `UDEModel` somut parametreleştirme; dummy NN’yi 0-delikte kaldırma.

---

## 12. LONG-TERM VISION COMPATIBILITY

`docs/research/BIO-DYNAX-VISION.md` bir v1.0 charter’ıdır; implementasyon değildir. Mevcut ağaç onun **1. evresi** (forensic audit) için girdi, çıktısı değil.

**Zaten temel olan:** bilinen/bilinmeyen ayrımı; derlenmiş dinamik; tek delik kavramı (yolda, derleyicide değil); graph-local kütüphane; constrained UDE (`D≥0`, `P−D·u`); çoklu IC eğitimi; pratik (parametre) identifiability uyarısı; aday + retcode; provenance tohumları; katmanlı CI (fast vs recovery); iddia dürüstlüğü.

**Güçlendirilmesi gereken / eksik:** vizyon Q2–Q7’nin ayrımı (bugün Q1 fit ile Q5 destek karışır); **fonksiyonel** identifiability; mekanizma-öncelikli metrik (held-out); çoklu aday hipotez nesnesi; çok tohum + gürültü + yoğunluk zarfı; ciddi baseline; out-of-sample mekanizma; belirsizlik kalibrasyonu (Fisher aralıkları var, NN/`D` için yok); bilimsel rapor nesnesi (bugün NamedTuple + stdout); `v1_contract.md`; performans/derleme vizyonunun ötesinde allocation kapıları.

**Uyumsuzluk:** vizyon \(\dot x = f_{\mathrm{known}}(x,\theta)+D(z;\phi)\) yazar; kod \(P-D\cdot u\). “Unknown hole” vizyonda genel eksik mekanizma, kodda yalnızca yıkım. Unique-claim **belirlenemezliği başarı sayar**; vizyon “identifiable across ICs” ister. Bu, evrimin üzerine inşa değil, sözleşme değişikliği gerektirir.

---

## 13. TOP 10 PRIORITIES

1. **Held-out mekanizma protokolü** — eğitimde görülmeyen IC ve `r` aralığında `D` ve yörünge; residual 0.30’u gerekçelendir veya sıkılaştır.
2. **Fonksiyonel identifiability tanısı** — çoklu eğitim `D`’leri; yörünge anlaşması vs fonksiyon anlaşmazlığı; yapısal sertifika diye adlandırma.
3. **Çok tohum UDE kapısı** — median/başarı oranı; tek seed 103 iskelet olarak kalsın, v1.0 kapısı olmasın.
4. **Graph-prior’u eğitilmiş `D`’ye taşı** — analitik Hill ablation’ı sızıntı olarak etiketle.
5. **Sembolik eşdeğerlik** — extras’ın fonksiyonel etkisi; kanonik forma indirgeme ayrı, kapalı kalsın.
6. **Aynı-görev baseline** — aynı örnekler, kütüphane, metrik; DDS skip’ini kazanım yazma (zaten yazılmıyor; karşılaştırma hâlâ yok).
7. **`run_recovery_suite` ve honesty dosyalarını ayır** — bilim fonksiyonları vs string kilitleri; testleri buna göre etiketle.
8. **Reproduction artifact** — Manifest veya lock + seed + ham metrik + commit; unique-claim satırını yeniden üret.
9. **Public API vs `BioDynaX.foo`** — v1.0 öncesi donmuş yüzey; `UDEModel` tip kararlılığı.
10. **v1.0 bilimsel sözleşme belgesi** — `P−D·u`, tek yıkım deliği, “başarı = recall + residual + belirlenemez ölçek”; vizyon denklemini sessizce miras alma.

---

**Özet yargı.** BioDynaX.jl, 2-durumlu sentetik Hill fikstüründe, bilinen grafta, tam bir nöral yıkım deliğinde, tek tohumda: yörüngeyi uydurabildiğini, `D` destekte `r^2`’yi hatırladığını, ekstra `1`/`r` monomial bıraktığını ve `k_prod` ile `D` ölçeğinin pratik olarak koliner olduğunu **gösterir**. Kanonik mekanizma kurtarma, fonksiyonel identifiability, çok rejim genellemesi veya harici yönteme üstünlük **gösterilmemiştir**. Deponun en olgun parçası bilimsel iddiayı **daraltan** kilitlerdir; en zayıf parçası o dar iddianın bile tek fikstüre, gevşek residual eşiğine ve analitik-keşif sızıntısına dayanmasıdır.