# Rekonstrukcija digitalne slike iz degradiranih i niskorezolucionih podataka primenom interpolacije i neuronske mreže

## 1. Opis projekta

Problem sa kojim se susrećemo kod digitalnih slika je taj što često nisu
dostupne u idealnom, tj. željenom obliku. Prenos ili skladištenje
podataka mogu dovesti do smanjenja rezolucije slike ili gubitka
piksela. Takve degradacije direktno utiču na kvalitet prikaza i dalju
obradu slike.

Cilj ovog projekta je rekonstrukcija digitalne slike korišćenjem
numeričkih interpolacionih metoda. Rekonstrukcija se posmatra kao
problem procene izgubljenih piksela na osnovu poznatih piksela. U
radu se razmatraju dva osnovna problema, a to su povećanje rezolucije
slike (upscaling) i nadoknada nedostajućih piksela. Poseban akcenat
stavljen je na poređenje različitih interpolacionih metoda, kao i na
poređenje rekonstrukcije slike sa i bez primene neuronske mreže
(SRCNN), uz analizu njihovog uticaja na kvalitet rekonstruisane slike.

Lagranžova interpolacija se u ovom projektu koristi kao teorijska
osnova za razumevanje polinomskih interpolacionih metoda, koje
primenjujemo za obradu digitalnih slika.

Sama rekonstrukcija slike se ostvaruje korišćenjem lokalnih i
numerički stabilnijih interpolacionih tehnika, kao što su bilinearna
interpolacija, bikubna interpolacija i interpolacija splajnovima, pri
čemu korisnik ima mogućnost da dodatno primeni neuronsku mrežu
radi unapređivanja kvaliteta rekonstruisane slike.

Sve navedene numeričke interpolacione metode su implementirane ručno,
bez korišćenja ugrađenih funkcija biblioteka za obradu slike, u cilju detaljne
analize njihovih numeričkih osobina i uticaja na kvalitet rekonstruisane slike.

## 2. Podaci

Za trening neuronske mreže koristi se skup od **800 slika** (DIV2K dataset),
smeštenih u `data/train/original/`. Degradirane (niskorezolucione) verzije
generišu se automatski bicubic downsampling-om sa faktorom 3
(`data/train/degraded/`).

Za evaluaciju i benchmark koristi se skup od **14 klasičnih test slika**
(Set14: baboon, barbara, bridge, coastguard, comic, face, flowers, foreman,
lenna, man, monarch, pepper, ppt3, zebra), smeštenih u
`data/input/original/` i `data/input/degraded/`.

U korisničkom režimu rada, korisnik za ulazne podatke prosleđuje
digitalnu sliku u RGB formatu (PNG ili JPEG) putem foldera `user/input/`.
Ulazna slika se posmatra kao već degradirana, odnosno niskorezoluciona
ili sa nedostajućim pikselima.

Slike se u pozadini programa konvertuju iz RGB u **YCbCr** prostor boja:

$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

$$C_b = -0.169 \cdot R - 0.331 \cdot G + 0.500 \cdot B + 128$$

$$C_r = 0.500 \cdot R - 0.419 \cdot G - 0.081 \cdot B + 128$$

Interpolacione metode se primenjuju nezavisno nad svakim kanalom (Y, Cb, Cr),
dok se neuronska mreža primenjuje isključivo nad **Y kanalom** (luminansom),
jer ljudsko oko je najosetljivije na promene u osvetljenosti.

## 3. Metodologija

Projekat je organizovan modularno, iz dva osnovna funkcionalna dela:
interpolacione metode (3.3) i neuronska mreža (3.4), pri čemu korisnik
putem jednostavnog menija bira željenu operaciju.

### 3.1. Rekonstrukcija nedostajućih piksela

Interpolacija se koristi za procenu vrednosti piksela koji nisu
poznati u ulaznoj slici. Za svaki nedostajući piksel razmatra se
njegova lokalna okolina, na osnovu koje se procenjuje vrednost
interpolacionom metodom (samim tim, korišćenjem više
interpolacionih metoda, postoji više izlaza). Posebna pažnja
posvećena je obradi ivica slike i slučajevima sa ograničenim
brojem dostupnih susednih piksela.

### 3.2. Rekonstrukcija slike u višoj rezoluciji

Povećanje rezolucije slike posmatra se kao problem
rekonstrukcije nedostajućih uzoraka između postojećih piksela.
Interpolacija se primenjuje separabilno — prvo po jednoj osi, zatim po
drugoj — nezavisno nad svakim YCbCr kanalom, nakon čega se kanali
spajaju u rekonstruisanu RGB sliku.

### 3.3. Interpolacione metode

U okviru projekta implementirano je više interpolacionih
metoda koje se koriste, kako za rekonstrukciju nedostajućih
piksela, tako i za povećanje rezolucije slike.

Interpolacija u okviru projekta primenjuje se
lokalno, nad ograničenom okolinom piksela. Za bilinearnu
interpolaciju koristi se okolina 2×2, za bikubnu interpolaciju 4×4,
dok se kod splajn interpolacije razmatraju svi pikseli u datom
redu/koloni. Na ivicama slike koristi se prilagođena okolina
(refleksija indeksa), kako bi se izbegao izlazak van granica slike.

#### 3.3.1. Lagranžova interpolacija

Lagranžova interpolacija koristi se kao teorijska osnova
za razumevanje polinomske interpolacije. Metoda
konstruiše interpolacioni polinom koji tačno prolazi
kroz poznate tačke. Interpolacioni polinom se definiše u
obliku:

$$P(x) = \sum_{i=0}^{n} y_i \cdot L_i(x)$$

gde su $L_i(x)$ Lagranžovi bazni polinomi definisani kao:

$$L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}$$

Osnovna osobina ovih polinoma je da važi $L_i(x_i)=1$,
dok je $L_i(x_j)=0$ za sve $j \neq i$, čime se obezbeđuje da
interpolacioni polinom u svakoj tački $x_i$ poprima tačnu
vrednost $y_i$.

U kontekstu obrade digitalnih slika, Lagranžova
interpolacija može se posmatrati kao metod za procenu
vrednosti nepoznatih piksela na osnovu poznatih
uzoraka.

Zbog numeričke nestabilnosti i računske složenosti pri
radu sa većim brojem tačaka, Lagranžova interpolacija
se ne koristi direktno u rekonstrukciji slike, već služi kao
referentni teorijski model za ostale interpolacione
metode.

Metode poput bilinearne, bikubne interpolacije i
interpolacije splajnovima mogu se posmatrati kao
lokalne polinomske interpolacije, koje izbegavaju
numeričke probleme globalne Lagranžove interpolacije.

#### 3.3.2. Bilinearna interpolacija

Bilinearna interpolacija predstavlja lokalnu
interpolacionu metodu koja koristi četiri najbliža piksela
u okolini 2×2 nepoznate tačke.

Za tačku $(x, y)$ koja se nalazi između četiri poznata piksela
$Q_{00}, Q_{01}, Q_{10}, Q_{11}$, sa frakcionalnim pomerajima
$t_x = x - \lfloor x \rfloor$ i $t_y = y - \lfloor y \rfloor$, bilinearna interpolacija
se definiše kao:

$$f(x, y) = (1 - t_y)(1 - t_x) \cdot Q_{00} + t_y(1 - t_x) \cdot Q_{10} + (1 - t_y) \cdot t_x \cdot Q_{01} + t_y \cdot t_x \cdot Q_{11}$$

Prvo se vrši linearna interpolacija u horizontalnom
pravcu (po $x$), a zatim u vertikalnom (po $y$). U slučaju
rekonstrukcije nedostajućih piksela, koristi se iterativno
distancom ponderisano popunjavanje u okolini 2×2.

#### 3.3.3. Bikubna interpolacija

Bikubna interpolacija koristi širu lokalnu okolinu 4×4 od
bilinearne i zasniva se na kubnoj interpolaciji u oba
prostorna pravca. Implementirana je kao **separabilna**
interpolacija sa **Catmull-Rom** jezgrom (parametar $a = -0.5$).

Za frakcionalni pomeraj $t \in [0, 1)$, težine četiri susedna piksela
na pozicijama $-1, 0, 1, 2$ računaju se kao:

$$w_0 = -0.5t + t^2 - 0.5t^3$$

$$w_1 = 1 - 2.5t^2 + 1.5t^3$$

$$w_2 = 0.5t + 2t^2 - 1.5t^3$$

$$w_3 = -0.5t^2 + 0.5t^3$$

Interpolacija se primenjuje separabilno: prvo se vrši kubna interpolacija duž
horizontalne ose (po $x$), a zatim se rezultat interpolira duž vertikalne ose (po $y$).
Za obradu ivica koristi se **refleksija indeksa** ($\text{reflect}(i, N)$), čime se
izbegava izlazak van dimenzija slike.

U odnosu na bilinearnu interpolaciju, bikubna metoda obezbeđuje
glađe prelaze i bolje očuvanje ivica.

#### 3.3.4. Interpolacija kubnim splajnovima

Interpolacija splajnovima zasniva se na korišćenju
parcijalnih polinoma trećeg stepena, definisanih nad
manjim intervalima. Ovakav pristup omogućava veću
numeričku stabilnost i smanjenje oscilacija, u poređenju sa
globalnom polinomskom interpolacijom.

Za $n$ poznatih čvorova $(x_0, y_0), \ldots, (x_{n-1}, y_{n-1})$ i query tačku $x_q$ koja
pripada segmentu $[x_i, x_{i+1}]$, natural cubic spline se evaluira kao:

$$S(x_q) = m_i \cdot \frac{(x_{i+1} - x_q)^3 - (x_{i+1} - x_q)}{6} \cdot h_i + m_{i+1} \cdot \frac{(x_q - x_i)^3 - (x_q - x_i)}{6} \cdot h_i + y_i \cdot \frac{x_{i+1} - x_q}{h_i} + y_{i+1} \cdot \frac{x_q - x_i}{h_i}$$

gde je $h_i = x_{i+1} - x_i$, a $m_i$ su drugi izvodi splajn funkcije u čvorovima,
koji se dobijaju rešavanjem **tridiagonalnog sistema** primenom
**Thomas-ovog algoritma**:

$$h_{i-1} \cdot m_{i-1} + 2(h_{i-1} + h_i) \cdot m_i + h_i \cdot m_{i+1} = 6\left(\frac{y_{i+1} - y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}}\right)$$

sa graničnim uslovima $m_0 = m_{n-1} = 0$ (natural spline).

Interpolacija se primenjuje separabilno — prvo se splajn evaluira duž
horizontalne ose za svaki red slike, a zatim duž vertikalne ose za
svaku kolonu rezultata.

### 3.4. Neuronska mreža (SRCNN)

Neuronska mreža korišćena u projektu zasniva se na **SRCNN**
(Super-Resolution Convolutional Neural Network) arhitekturi,
koja predstavlja konvolucionu neuronsku mrežu
namenjenu rešavanju problema super-rezolucije slika. U okviru
projekta, neuronska mreža koristi se kao dodatni korak nakon
klasične interpolacije, pri čemu interpolisana slika (Y kanal)
predstavlja ulaz mreže. Na ovaj način, neuronska mreža uči da
koriguje greške koje nastaju primenom interpolacionih metoda i da
poboljša kvalitet rekonstruisane slike.

#### Arhitektura

SRCNN se sastoji od tri uzastopna konvoluciona sloja sa rezidualnom
(preskočnom) vezom:

| Sloj  | Kernel | Kanali  | Aktivacija | Opis                               |
| ----- | ------ | ------- | ---------- | ---------------------------------- |
| Conv1 | 9×9    | 1 → 64  | ReLU       | Ekstrakcija lokalnih osobina       |
| Conv2 | 5×5    | 64 → 32 | ReLU       | Nelinearno preslikavanje           |
| Conv3 | 5×5    | 32 → 1  | Nema       | Rekonstrukcija rezidualnog signala |

Izlaz mreže je definisan kao:

$$\hat{Y} = \text{clamp}\!\left(X + f(X),\ 0,\ 1\right)$$

gde je $X$ ulazna interpolisana slika (Y kanal), a $f(X)$ izlaz konvolucionih
slojeva (rezidual — naučena korekcija). Funkcija $\text{clamp}$ ograničava
vrednosti piksela na opseg $[0, 1]$.

**ReLU** (Rectified Linear Unit) aktivaciona funkcija primenjuje se nakon
prvog i drugog konvolucionog sloja:

$$\text{ReLU}(x) = \max(0, x)$$

Treći sloj nema aktivacionu funkciju — izlaz je linearan jer rezidual može
biti i pozitivan (posvetliti piksel) i negativan (potamniti piksel).

#### Inicijalizacija težina

Težine prvog i drugog sloja inicijalizuju se **Kaiming Normal**
inicijalizacijom, optimizovanom za ReLU:

$$W \sim \mathcal{N}\!\left(0,\ \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

gde je $n_{\text{in}} = C_{\text{in}} \cdot k_h \cdot k_w$ (broj ulaznih veza po neuronu).

Treći sloj se inicijalizuje na **nulu** ($W_3 = 0$, $b_3 = 0$), čime se
obezbeđuje da na početku treninga $f(X) = 0$, pa je izlaz mreže jednak
ulazu (identitet). Time se sprečava da netrenirani model pogorša sliku.

#### Loss funkcija — MSE

Proces treniranja mreže zasniva se na minimizaciji **MSE** (Mean Squared Error)
greške između izlaza neuronske mreže i referentne slike visoke rezolucije:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \left(\hat{y}_i - y_i\right)^2$$

gde su $\hat{y}_i$ predviđene vrednosti piksela, $y_i$ stvarne vrednosti iz
originalne HR slike, a $N$ ukupan broj piksela u patch-u.

#### Optimizacija — Adam

Za ažuriranje parametara koristi se **Adam** (Adaptive Moment Estimation)
optimizer, koji za svaki parametar $\theta$ održava eksponencijalni prosek
gradijenata ($m_t$) i kvadrata gradijenata ($v_t$):

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

gde je $\eta$ learning rate, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

Koristi se **diferencijalni learning rate**: Conv1 i Conv2 uče sa
$\eta = 10^{-4}$, dok Conv3 uči 10× sporije ($\eta = 10^{-5}$), jer je izlazni
sloj osetljiviji na promene parametara.

**ReduceLROnPlateau** scheduler prepolovljava learning rate (faktor 0.5) ako se
validation loss ne poboljša 3 uzastopne epohe.

#### Backward propagation

Gradijenti se računaju automatski pozivom `loss.backward()` primenom
**chain rule** (pravilo lanca diferenciranja). Dodatno, primenjuje se
**gradient clipping** ($\text{max\_norm} = 1.0$) koji sprečava eksploziju gradijenata.

Rezidualna veza omogućava direktan protok gradijenta od izlaza do ulaza,
čime se ublažava problem nestajućih gradijenata.

#### Regularizacija

- **Early stopping** — trening se prekida ako se validation loss ne poboljša
  10 uzastopnih epoha
- **Data augmentation** — horizontalni/vertikalni flip i rotacije (0°, 90°, 180°, 270°)
  nasumično se primenjuju na trening patch-eve

## 4. Evaluacija rezultata

Kvalitet rekonstruisane slike evaluira se korišćenjem **PSNR** i **MSE** metrika,
koje predstavljaju numeričke mere odstupanja između originalne i
rekonstruisane slike.

**MSE** (Mean Squared Error):

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2$$

**PSNR** (Peak Signal-to-Noise Ratio):

$$\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{MAX^2}{\text{MSE}}\right) = 20 \cdot \log_{10}\!\left(\frac{MAX}{\sqrt{\text{MSE}}}\right)$$

gde je $MAX = 255$ maksimalna vrednost piksela za 8-bitne slike.
Veći PSNR znači bolji kvalitet rekonstrukcije.

Kvantitativna evaluacija kvaliteta rekonstrukcije vrši se isključivo pri
testiranju, gde se kao referenca koristi originalna slika.
Niskorezoluciona ulazna slika dobija se veštačkim degradiranjem
originala (bicubic downsampling sa faktorom 3), isključivo u svrhu
evaluacije algoritama, nakon čega se primenjuje rekonstrukcija.
U korisničkom režimu rada, gde referentna slika nije dostupna,
rezultati se analiziraju isključivo vizuelno.

## 5. Struktura projekta

```
src/
  interpolation/
    bilinear.py      – ručna bilinearna interpolacija
    bicubic.py       – ručna bikubna interpolacija (Catmull-Rom)
    spline.py        – ručna kubna spline interpolacija (Thomas algoritam)
  neural_network/
    srcnn_model.py   – SRCNN arhitektura (Y kanal, rezidualna veza)
    dataset.py       – LR/HR dataset sa keširanjem i patch ekstrakcijom
    train.py         – trening sa train/val split-om i early stopping-om
  evaluation/
    metrics.py       – PSNR i MSE metrike
  utils/
    image_io.py      – učitavanje, čuvanje, RGB↔YCbCr konverzija
    degradation.py   – generisanje degradiranih slika (downsampling)
  main.py            – interaktivni meni za inference
scripts/
  generate_degraded.py   – generisanje LR slika iz originala
  benchmark_methods.py   – automatsko poređenje svih metoda
data/
  train/original/        – HR trening slike (800 slika)
  train/degraded/        – LR trening slike (generisane)
  input/original/        – HR test slike (14 slika, za evaluaciju)
  input/degraded/        – LR test slike (generisane)
user/
  input/                 – korisnikov ulaz
  output/                – rekonstruisane slike
```

## 6. Tehnologije

- **Python** — programski jezik
- **NumPy** — numeričke operacije (matrice, vektorizacija)
- **Pillow** — isključivo za učitavanje i čuvanje slika
- **PyTorch** — implementacija i trening SRCNN neuronske mreže
- **SciPy** — uniform filter za evaluacionu metriku

Biblioteka za obradu slike (Pillow) koristi se isključivo za
učitavanje i čuvanje slika, dok su sve interpolacione metode
implementirane ručno.

## 7. Pokretanje

### 1) Generisanje degradiranih slika

```bash
python scripts/generate_degraded.py
```

Generisanje trening LR skupa:

```bash
python scripts/generate_degraded.py --input-dir data/train/original --output-dir data/train/degraded --scale 3
```

### 2) Trening SRCNN

```bash
python -m src.neural_network.train --epochs 50 --scale 3 --patch 33 --batch-size 64 --learning-rate 1e-4 --patience 10
```

Najbitniji argumenti:

- `--batch-size` — veličina mini-batch-a
- `--learning-rate` — početni learning rate ($10^{-4}$)
- `--patience` — broj epoha za early stopping
- `--max-samples` — maksimalan broj slika za trening (default: 800)

Izlazi modela:

- `models/srcnn_y.pth` — poslednja epoha
- `models/srcnn_y_best.pth` — najbolji validation loss

### 3) Ručni inference (jedna slika)

Staviti sliku u `user/input/input_image.png`, zatim:

```bash
python -m src.main
```

Meni podržava:

1. Bicubic
2. Bilinear
3. Spline
4. Bicubic + SRCNN
5. Spline + SRCNN
6. Bilinear + SRCNN

### 4) Benchmark svih metoda

```bash
python scripts/benchmark_methods.py --scale 3
```

CSV izveštaj se čuva u `user/output_recon/benchmark_results.csv`.

## 8. Podela rada

- **SV 73/2024 Danilo Torbica** — implementacija numeričkih
  interpolacionih metoda (bilinearna, bikubna i interpolacija kubnim
  splajnovima), rekonstrukcija nedostajućih piksela i povećanje
  rezolucije slike primenom navedenih metoda, testiranje i analiza
  rezultata.

- **SV 74/2024 Vladimir Petrović** — implementacija neuronske mreže za
  rekonstrukciju slike (SRCNN), treniranje, testiranje i evaluacija.

- **Zajednički rad** — integracija modula, implementacija glavnog
  programa sa menijem, poređenje rezultata klasičnih interpolacionih
  metoda i metoda zasnovanih na neuronskim mrežama, kao i izrada
  dokumentacije.

## 9. Literatura i resursi

- https://www.cambridgeincolour.com/tutorials/image-interpolation.htm
- https://www.youtube.com/watch?v=bzp_q7NDdd4
- https://en.wikipedia.org/wiki/Interpolation
- https://en.wikipedia.org/wiki/Bicubic_interpolation
- https://en.wikipedia.org/wiki/Spline_interpolation
- https://arxiv.org/abs/1501.00092
