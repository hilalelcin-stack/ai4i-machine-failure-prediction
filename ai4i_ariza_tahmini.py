# -*- coding: utf-8 -*-
"""
AI4I 2020 - Makine Ariza Tahmini (Bekir Hoca case calismasi)
Hilal Elcin Sahin - MAKU Veri Bilimi ve Analitigi

Calistirma : python ai4i_ariza_tahmini.py
Gerekenler : pandas, numpy, scikit-learn, scipy, matplotlib   (requirements.txt)
Veri       : data/ai4i2020.csv  (UCI dataset 601; yoksa betigin yanindaki ai4i2020.csv)
Ciktilar   : outputs/  (tablolar .csv, grafikler .png, sonuclar.json, islenmis veri)

Bolum basliklari rapordaki etiketlerle aynidir: SORU 1a, 1b, 1c, 2a, 2b, 2c, 2d, 3a, 3b, 3c.

Dogrulama duzeni (butun kararlar buradan cikar):
  - %80 gelistirme / %20 test, stratified, seed 42. Test kumesi hicbir secimde kullanilmaz, yalnizca sonuc raporlanir.
  - Gelistirme kumesi icinde 5 katli StratifiedKFold.
  - Satir atan her temizlik adimi YALNIZCA egitim katlarina uygulanir; dogrulama katlari ve test
    eksiksiz kalir (temizlik dogrulamaya sizmaz).
  - Olcekleme ve veriden ogrenilen donusumler Pipeline icinde -> her katta yeniden fit edilir.
  - Hiperparametre secimi CV ile; secimden kaynaklanan iyimserligi olcmek icin ic ice (nested) CV.
  - Karar esigi yalnizca out-of-fold (OOF) tahminlerden, maliyet kuraliyla (varsayim: 1 FN = 10 FP).
  - Test sonuclari bootstrap %95 guven araliklariyla raporlanir.
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# =============================================================================
# 0. AYARLAR, VERI, BOLME
# =============================================================================
BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
GRAF = OUT / "grafikler"
GRAF.mkdir(parents=True, exist_ok=True)

SEED, TEST_SIZE, N_FOLDS = 42, 0.20, 5
TARGET = "Machine failure"
YASAK = ["TWF", "HDF", "PWF", "OSF", "RNF"]       # case: ariza SONUCU gostergeleri -> girdi degil
KIMLIK = ["UDI", "Product ID"]                     # case: kimlik -> girdi degil
HAM = ["Type", "Air temperature [K]", "Process temperature [K]",
       "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
SAY_HAM = HAM[1:]
MODELLER = ["Logistic Regression", "KNN", "Decision Tree"]

# SORU 3a karar kurali (VARSAYIM, gercek maliyet verisi yok): 1 kacirilan ariza (FN) = 10 yanlis
# alarm (FP). Esik yalnizca OOF tahminlerinden, bu maliyeti en aza indirecek sekilde secilir.
MALIYET_ORANI = 10

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
R = {}   # rapora giden butun sayilar -> outputs/sonuclar.json


def baslik(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


yol = BASE / "data" / "ai4i2020.csv"
if not yol.exists():
    yol = BASE / "ai4i2020.csv"
df = pd.read_csv(yol)
df.columns = [c.replace("﻿", "").strip() for c in df.columns]
y = df[TARGET].to_numpy()

tr_idx, te_idx = train_test_split(df.index.to_numpy(), test_size=TEST_SIZE,
                                  random_state=SEED, stratify=y)
tr_idx, te_idx = np.sort(tr_idx), np.sort(te_idx)
print(f"Veri {df.shape} | gelistirme {len(tr_idx)} (ariza {y[tr_idx].sum()}) | "
      f"test {len(te_idx)} (ariza {y[te_idx].sum()})")

# =============================================================================
# ORTAK ARACLAR: ozellikler, modeller, CV
# =============================================================================
SIRA = {"L": 0, "M": 1, "H": 2}   # urun kalitesi sirali bir derece


def tablo(sirali=True):
    """Ham 6 girdi + aday ozellikler. Hicbiri hedefi kullanmaz; hepsi olcum aninda
    sensor degerlerinden hesaplanabilir."""
    X = df[HAM].copy()
    X["Type"] = X["Type"].map(SIRA) if sirali else X["Type"].astype("category").cat.codes
    X["Sicaklik farki [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    X["Guc [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    X["Asinma x Tork"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    X["Asinma orani"] = df["Tool wear [min]"] / 200.0
    X["Tork x Devir"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    X["Isi yuku"] = X["Guc [W]"] / X["Sicaklik farki [K]"]
    return X


class GucSapmasi(BaseEstimator, TransformerMixin):
    """Normal calisma bandindan sapma: z = (tork x devir - medyan) / std  ->  z^2.
    Arizalar gucun IKI UCUNDA toplaniyor (U sekli); dogrusal model bunu ham degerden
    ogrenemez. Medyan/std yalnizca fit edilen (egitim) veriden ogrenilir."""
    AD = "Guc sapmasi kare"

    def fit(self, X, y=None):
        p = X["Torque [Nm]"] * X["Rotational speed [rpm]"]
        self.m_, self.s_ = float(p.median()), float(p.std())
        return self

    def transform(self, X):
        X = X.copy()
        p = X["Torque [Nm]"] * X["Rotational speed [rpm]"]
        X[self.AD] = ((p - self.m_) / self.s_) ** 2
        return X


def model_kur(ad, sayisal, sapma=False, **ayar):
    """Zorunlu uc model (+ Dummy taban). LR ve KNN icin olcekleme Pipeline icinde
    (sutunlar K, rpm, Nm gibi farkli birimlerde; KNN mesafe tabanli). Agac olceklemeye duyarsiz."""
    if ad == "Dummy":
        clf = DummyClassifier(strategy="prior")
    elif ad == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
    elif ad == "KNN":
        clf = KNeighborsClassifier()
    else:
        clf = DecisionTreeClassifier(random_state=SEED)
    clf.set_params(**ayar)
    say = list(sayisal) + ([GucSapmasi.AD] if sapma else [])
    olcek = StandardScaler() if ad in ("Logistic Regression", "KNN") else "passthrough"
    ct = ColumnTransformer([("num", olcek, say), ("cat", "passthrough", ["Type"])])
    adimlar = ([("sapma", GucSapmasi())] if sapma else []) + [("prep", ct), ("clf", clf)]
    return Pipeline(adimlar)


SKF = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)


def cv_skor(model, X, idx, at=None, oof=False, cv=SKF):
    """5 katli CV PR-AUC. `at`: egitim KATLARINDAN atilacak satirlar (bool maske).
    Dogrulama katlari her zaman eksiksiz -> temizlik dogrulama skoruna sizmaz."""
    Xd, yd = X.loc[idx], y[idx]
    skor, p_oof = [], np.zeros(len(idx))
    for a, b in cv.split(Xd, yd):
        a_idx = idx[a]
        if at is not None:
            a_idx = a_idx[~at[a_idx]]
        m = clone(model).fit(X.loc[a_idx], y[a_idx])
        p = m.predict_proba(Xd.iloc[b])[:, 1]
        p_oof[b] = p
        skor.append(average_precision_score(yd[b], p))
    skor = np.array(skor)
    return (skor, p_oof) if oof else skor


def test_olc(model, X, egit_idx, esik=0.5):
    """Modeli egitim satirlarinda egitir, KILITLI test kumesinde bir kez olcer."""
    m = clone(model).fit(X.loc[egit_idx], y[egit_idx])
    p = m.predict_proba(X.loc[te_idx])[:, 1]
    yt, yp = y[te_idx], (p >= esik).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return m, p, {"Accuracy": accuracy_score(yt, yp),
                  "Precision": precision_score(yt, yp, zero_division=0),
                  "Recall": recall_score(yt, yp, zero_division=0),
                  "F1": f1_score(yt, yp, zero_division=0),
                  "PR-AUC": average_precision_score(yt, p),
                  "ROC-AUC": roc_auc_score(yt, p),
                  "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def karmasa(yt, p, t):
    tn, fp, fn, tp = confusion_matrix(yt, (p >= t).astype(int), labels=[0, 1]).ravel()
    return int(tp), int(fp), int(fn), int(tn)


# =============================================================================
# SORU 1a - Hedefin dagilimi ve siniflandirma zorlugu
# =============================================================================
baslik("SORU 1a - Machine failure dagilimi")
vc = df[TARGET].value_counts().sort_index()
tip_oran = (df.groupby("Type")[TARGET].mean() * 100).reindex(["L", "M", "H"])
R["1a"] = {"saglam": int(vc[0]), "ariza": int(vc[1]), "ariza_oran": float(vc[1] / len(df)),
           "dengesizlik": float(vc[0] / vc[1]), "dummy_acc": float(vc[0] / len(df)),
           "tip_oran": tip_oran.round(2).to_dict(),
           "tip_sayi": df["Type"].value_counts().reindex(["L", "M", "H"]).to_dict()}
print(vc.to_string())
print(f"Ariza orani %{100*R['1a']['ariza_oran']:.2f} | dengesizlik 1:{R['1a']['dengesizlik']:.1f} | "
      f"herkese 'saglam' diyen kuralin accuracy'si {R['1a']['dummy_acc']:.4f}")
print("Urun tipine gore ariza orani (%):", R["1a"]["tip_oran"])

# =============================================================================
# SORU 1b - Arizali ve arizasiz makinelerin sayisal karsilastirmasi
# =============================================================================
baslik("SORU 1b - Arizali vs arizasiz karsilastirma")
satir = []
for c in SAY_HAM:
    a, b = df.loc[y == 1, c], df.loc[y == 0, c]
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    satir.append({"Degisken": c, "Arizasiz ort": b.mean(), "Arizali ort": a.mean(),
                  "Arizasiz medyan": b.median(), "Arizali medyan": a.median(),
                  "Arizasiz std": b.std(), "Arizali std": a.std(),
                  "Cohen d": (a.mean() - b.mean()) / sp,
                  "Mann-Whitney p": stats.mannwhitneyu(a, b).pvalue})
t1b = pd.DataFrame(satir).sort_values("Cohen d", key=abs, ascending=False)
t1b.round(4).to_csv(OUT / "s1b_karsilastirma.csv", index=False)
print(t1b.round(3).to_string(index=False))
R["1b"] = t1b.round(4).to_dict("records")

# uretim yonetimi icin bant tablolari: hangi kosulda ariza siklasiyor?
bantlar = {
    "Tool wear [min]": [-1, 100, 200, 260],
    "Torque [Nm]": [0, 30, 50, 60, 80],
    "Rotational speed [rpm]": [0, 1380, 1600, 2000, 3000],
}
R["1b_bant"] = {}
for c, kes in bantlar.items():
    g = df.groupby(pd.cut(df[c], kes), observed=True)[TARGET].agg(["count", "mean"])
    g["mean"] = (g["mean"] * 100).round(2)
    R["1b_bant"][c] = [{"bant": str(i), "n": int(r["count"]), "ariza_yuzde": float(r["mean"])}
                       for i, r in g.iterrows()]
    print(f"\n{c} bandina gore ariza orani (%):")
    print(g.rename(columns={"count": "kayit", "mean": "ariza %"}).to_string())

# =============================================================================
# SORU 1c - Hangi degiskenler tahmin icin daha yararli?
# =============================================================================
baslik("SORU 1c - Tek degiskenli ayirt edicilik (ROC-AUC)")
Xf = tablo(True)
aday = SAY_HAM + ["Sicaklik farki [K]", "Tork x Devir", "Asinma x Tork"]
tekli = []
for c in aday:
    a = roc_auc_score(y, Xf[c])
    tekli.append({"Degisken": c, "ROC-AUC": max(a, 1 - a),
                  "Yon": "yuksek=riskli" if a >= .5 else "dusuk=riskli",
                  "Turetilmis": c not in SAY_HAM})
z = (Xf["Tork x Devir"] - Xf["Tork x Devir"].median()) / Xf["Tork x Devir"].std()
tekli.append({"Degisken": "Guc sapmasi kare", "ROC-AUC": roc_auc_score(y, z ** 2),
              "Yon": "bant disi=riskli", "Turetilmis": True})
tekli = pd.DataFrame(tekli).sort_values("ROC-AUC", ascending=False)
tekli.round(4).to_csv(OUT / "s1c_tekli_ayirt_edicilik.csv", index=False)
print(tekli.round(4).to_string(index=False))
R["1c"] = tekli.round(4).to_dict("records")

# --- Soru 1 grafigi (rapor sayfa 1)
fig, ax = plt.subplots(1, 4, figsize=(13, 2.9), gridspec_kw={"width_ratios": [0.8, 1, 1, 1.15]})
ax[0].bar(["Arizasiz", "Arizali"], vc.values, color=["#5B8FF9", "#E8684A"])
for i, v in enumerate(vc.values):
    ax[0].text(i, v + 200, f"{v}\n%{100*v/len(df):.1f}", ha="center", fontsize=7.5)
ax[0].set_ylim(0, 11800); ax[0].set_title("1a) Hedef dagilimi")
for k, c in enumerate(["Torque [Nm]", "Tool wear [min]"]):
    veri_k = [df.loc[y == 0, c], df.loc[y == 1, c]]
    try:   # matplotlib >= 3.9
        ax[k + 1].boxplot(veri_k, tick_labels=["Arizasiz", "Arizali"], widths=.55,
                          flierprops={"markersize": 2, "alpha": .3})
    except TypeError:   # eski matplotlib
        ax[k + 1].boxplot(veri_k, labels=["Arizasiz", "Arizali"], widths=.55,
                          flierprops={"markersize": 2, "alpha": .3})
    ax[k + 1].set_title(f"1b) {c}")
m0, m1 = y == 0, y == 1
ax[3].scatter(df.loc[m0, "Rotational speed [rpm]"], df.loc[m0, "Torque [Nm]"], s=2, alpha=.2,
              c="#9DB4D6", label="arizasiz")
ax[3].scatter(df.loc[m1, "Rotational speed [rpm]"], df.loc[m1, "Torque [Nm]"], s=7, alpha=.85,
              c="#E8684A", label="arizali")
ax[3].set_xlabel("Rotational speed [rpm]"); ax[3].set_ylabel("Torque [Nm]")
ax[3].set_title("1b-1c) Arizalar guc bandinin iki ucunda"); ax[3].legend(fontsize=7, markerscale=2)
plt.tight_layout(); plt.savefig(GRAF / "soru1.png", dpi=170); plt.close()

# =============================================================================
# SORU 2a - Bolme ve on isleme (temizlik adimlari CV ile, yalnizca egitim katlarina)
# =============================================================================
baslik("SORU 2a - Temizlik adimlarinin CV katkisi (dogrulama katlari eksiksiz)")
herhangi = df[["TWF", "HDF", "PWF", "OSF"]].max(axis=1).to_numpy()
BELIRSIZ = (y == 1) & (herhangi == 0)            # ariza denmis, turu belirsiz
RASTGELE = (df["RNF"].to_numpy() == 1) & (y == 0)  # rastgele ariza hedefe sayilmamis
q999 = df.loc[tr_idx, ["Rotational speed [rpm]", "Torque [Nm]"]].quantile(.999)
AYKIRI = ((df["Rotational speed [rpm]"] > q999["Rotational speed [rpm]"]) |
          (df["Torque [Nm]"] > q999["Torque [Nm]"])).to_numpy()
YOK = np.zeros(len(df), bool)


def uc_model_cv(X, sayisal, at=None, sapma=False, ayar=None):
    ayar = ayar or {}
    return {m: cv_skor(model_kur(m, sayisal, sapma, **ayar.get(m, {})), X, tr_idx, at)
            for m in MODELLER}


def kabul_mu(yeni, eski, esik=0.0005, min_kat=9):
    """Kabul kurali: 3 modelin ortalama CV PR-AUC kazanci >= 0.0005 VE
    15 katin (3 model x 5) en az 9'unda iyilesme."""
    kazanc = np.mean([yeni[m].mean() for m in MODELLER]) - np.mean([eski[m].mean() for m in MODELLER])
    kat = int(sum((yeni[m] > eski[m]).sum() for m in MODELLER))
    return kazanc >= esik and kat >= min_kat, kazanc, kat


X_alfa, X_sira = tablo(False), tablo(True)
X_log = X_sira.copy(); X_log["Rotational speed [rpm]"] = np.log1p(X_log["Rotational speed [rpm]"])

adimlar = [("0) Ham (Type alfabetik kod)", X_alfa, YOK),
           ("1) Type sirali kod L<M<H", X_sira, YOK),
           ("2) + devir log1p", X_log, YOK),
           ("3) + belirsiz etiket egitimden atildi", X_sira, BELIRSIZ),
           ("4) + rastgele ariza (RNF) atildi", X_sira, BELIRSIZ | RASTGELE),
           ("5) + ust binde-1 aykiri atildi", X_sira, BELIRSIZ | AYKIRI)]
tem = {}
for ad, X, at in adimlar:
    tem[ad] = uc_model_cv(X, SAY_HAM, at)
    print(f"{ad:<40}" + " | ".join(f"{m.split()[0]} {tem[ad][m].mean():.4f}" for m in MODELLER)
          + f" || ort {np.mean([tem[ad][m].mean() for m in MODELLER]):.4f}")

ref = tem["1) Type sirali kod L<M<H"]
karar = {}
for ad in list(tem)[2:]:
    onceki = ref if not ad.startswith(("4", "5")) else tem["3) + belirsiz etiket egitimden atildi"]
    ok, kz, kt = kabul_mu(tem[ad], onceki)
    karar[ad] = {"kabul": bool(ok), "kazanc": float(kz), "iyilesen_kat": kt}
    print(f"  {ad:<40} kazanc {kz:+.4f}, iyilesen kat {kt}/15 -> {'KABUL' if ok else 'ELENDI'}")
LOG_KABUL = karar["2) + devir log1p"]["kabul"]
MASKE = YOK.copy()
if karar["3) + belirsiz etiket egitimden atildi"]["kabul"]:
    MASKE |= BELIRSIZ
    if karar["4) + rastgele ariza (RNF) atildi"]["kabul"]:
        MASKE |= RASTGELE
    if karar["5) + ust binde-1 aykiri atildi"]["kabul"]:
        MASKE |= AYKIRI
R["2a"] = {
    "egitim": int(len(tr_idx)), "test": int(len(te_idx)),
    "egitim_ariza": int(y[tr_idx].sum()), "test_ariza": int(y[te_idx].sum()),
    "belirsiz_toplam": int(BELIRSIZ.sum()), "belirsiz_egitim": int(BELIRSIZ[tr_idx].sum()),
    "rastgele_toplam": int(RASTGELE.sum()), "aykiri_egitim": int(AYKIRI[tr_idx].sum()),
    "adimlar": {ad: {m: float(v[m].mean()) for m in MODELLER} for ad, v in tem.items()},
    "karar": karar, "log_kabul": bool(LOG_KABUL),
    "maske_egitim": int(MASKE[tr_idx].sum()),
}
EGIT = tr_idx[~MASKE[tr_idx]]     # nihai modellerin egitildigi satirlar; test hic degismez
print(f"Egitimden atilan satir: {R['2a']['maske_egitim']} | nihai egitim {len(EGIT)} | test {len(te_idx)}")

# =============================================================================
# SORU 2a (devam) - Oznitelik paketleri, ileri secim (karar yalnizca CV)
# =============================================================================
baslik("SORU 2a - Oznitelik paketleri ve ileri secim")
X_tem = X_log if LOG_KABUL else X_sira
PAKET = {"P1 Isi (sicaklik farki)": (["Sicaklik farki [K]"], False),
         "P2 Guc [W]": (["Guc [W]"], False),
         "P3 Zorlanma (asinma x tork)": (["Asinma x Tork"], False),
         "P4 Asinma orani": (["Asinma orani"], False),
         "P5 Isi yuku + tork x devir": (["Isi yuku", "Tork x Devir"], False),
         "P6 Guc bandi sapmasi (z^2)": ([], True)}

taban = uc_model_cv(X_tem, SAY_HAM, MASKE)
tekil = {}
for ad, (sut, sap) in PAKET.items():
    tekil[ad] = uc_model_cv(X_tem, SAY_HAM + sut, MASKE, sapma=sap)
    _, kz, kt = kabul_mu(tekil[ad], taban)
    print(f"{ad:<30}" + " | ".join(f"{m.split()[0]} {tekil[ad][m].mean() - taban[m].mean():+.4f}"
                                   for m in MODELLER) + f" || ort {kz:+.4f}, kat {kt}/15")
sira = sorted(PAKET, key=lambda a: -np.mean([tekil[a][m].mean() for m in MODELLER]))
secili_sut, secili_sap, secili, guncel = list(SAY_HAM), False, [], taban
ileri = [{"Adim": "Taban (temizlenmis)", "Karar": "-",
          **{m: float(taban[m].mean()) for m in MODELLER}}]
for ad in sira:
    sut, sap = PAKET[ad]
    yeni = uc_model_cv(X_tem, secili_sut + sut, MASKE, sapma=secili_sap or sap)
    ok, kz, kt = kabul_mu(yeni, guncel)
    print(f"+ {ad:<28} ort kazanc {kz:+.4f}, kat {kt}/15 -> {'KABUL' if ok else 'ELENDI'}")
    ileri.append({"Adim": "+ " + ad, "Karar": "KABUL" if ok else "ELENDI",
                  **{m: float(yeni[m].mean()) for m in MODELLER},
                  "Ort kazanc": float(kz), "Iyilesen kat": kt})
    if ok:
        secili.append(ad); secili_sut += sut; secili_sap = secili_sap or sap; guncel = yeni
pd.DataFrame(ileri).round(4).to_csv(OUT / "s2a_ileri_secim.csv", index=False)
SAY_F, SAP_F = secili_sut, secili_sap
SUT_F = ["Type"] + SAY_F
print("Secilen paketler:", secili)
R["2a"]["paket_tekil"] = {a: {m: float(tekil[a][m].mean() - taban[m].mean()) for m in MODELLER}
                          for a in PAKET}
R["2a"]["ileri_secim"] = ileri
R["2a"]["secili"] = secili

# dokumandaki kesin ariza esikleri: denendi, dolayli hedef sizintisi oldugu icin ELENDI
X_esik = X_tem.copy()
X_esik["HDF bayragi"] = ((X_tem["Sicaklik farki [K]"] < 8.6) & (df["Rotational speed [rpm]"] < 1380)).astype(int)
X_esik["PWF bayragi"] = ((X_tem["Guc [W]"] < 3500) | (X_tem["Guc [W]"] > 9000)).astype(int)
X_esik["OSF bayragi"] = (X_tem["Asinma x Tork"] > df["Type"].map({"L": 11000, "M": 12000, "H": 13000})).astype(int)
esik_cv = uc_model_cv(X_esik, SAY_F + ["HDF bayragi", "PWF bayragi", "OSF bayragi"], MASKE, sapma=SAP_F)
R["2a"]["sizinti_esik_cv"] = {m: float(esik_cv[m].mean()) for m in MODELLER}
print("Kesin esik bayraklariyla CV (ELENDI - sizinti):", {m: round(v, 4) for m, v in R["2a"]["sizinti_esik_cv"].items()})

# =============================================================================
# SORU 2b - Uc modelin asamalara gore performansi + hiperparametre ayari
# =============================================================================
baslik("SORU 2b - Asama x model skor tablosu (test esigi 0,50)")
IZGARA = {"Logistic Regression": [{"C": c} for c in [0.01, 0.1, 1, 10, 100]],
          "KNN": [{"n_neighbors": k} for k in [5, 15, 25, 35, 51, 75]],
          "Decision Tree": [{"max_depth": d, "min_samples_leaf": l}
                            for d in [4, 5, 6, 7, 8, 10] for l in [1, 5, 10, 20, 30]]}


def izgara_sec(ad, X, sayisal, sapma, idx, at, cv=SKF):
    en = None
    for par in IZGARA[ad]:
        s = cv_skor(model_kur(ad, sayisal, sapma, **par), X, idx, at, cv=cv)
        if en is None or s.mean() > en[0].mean():
            en = (s, par)
    return en


AYAR = {}
for ad in MODELLER:
    s, par = izgara_sec(ad, X_tem, SAY_F, SAP_F, tr_idx, MASKE)
    AYAR[ad] = par
    kenar = par == IZGARA[ad][0] or par == IZGARA[ad][-1]
    print(f"  {ad:<20} en iyi {par} CV PR-AUC {s.mean():.4f} +- {s.std():.4f}"
          + ("  (izgara kenari!)" if kenar else ""))

asamalar = [("1 Temizlenmemis", X_alfa, SAY_HAM, False, YOK, {}),
            ("2 Temizlenmis", X_tem, SAY_HAM, False, MASKE, {}),
            ("3 Ozellikli", X_tem, SAY_F, SAP_F, MASKE, {}),
            ("4 Ozellikli + ayarli", X_tem, SAY_F, SAP_F, MASKE, AYAR)]
skor_satir, TEST_P, MODEL_F, OOF_F = [], {}, {}, {}
for asama, X, say, sap, at, ayar in asamalar:
    egit = tr_idx[~at[tr_idx]]
    for ad in ["Dummy"] + MODELLER:
        mdl = model_kur(ad, say, sap, **ayar.get(ad, {}))
        cvs, oof = cv_skor(mdl, X, tr_idx, at, oof=True)
        fit, p, met = test_olc(mdl, X, egit)
        skor_satir.append({"Asama": asama, "Model": ad, **met,
                           "CV PR-AUC ort": cvs.mean(), "CV PR-AUC sapma": cvs.std()})
        if asama.startswith("4"):
            TEST_P[ad], MODEL_F[ad], OOF_F[ad] = p, fit, oof
skor = pd.DataFrame(skor_satir)
skor.round(4).to_csv(OUT / "skorlar.csv", index=False)
print(skor[["Asama", "Model", "Accuracy", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC",
            "CV PR-AUC ort", "CV PR-AUC sapma"]].round(4).to_string(index=False))
R["2b"] = {"skorlar": skor.round(4).to_dict("records"), "ayar": {k: v for k, v in AYAR.items()}}

# --- ic ice (nested) CV: hiperparametre seciminin iyimserligini olc
baslik("DOGRULAMA - ic ice CV (secim yanliligi olmadan tahmini performans)")
dis = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + 1)
ic = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + 2)
R["nested"] = {}
for ad in MODELLER:
    dis_skor, secilen = [], []
    for a, b in dis.split(X_tem.loc[tr_idx], y[tr_idx]):
        a_idx, b_idx = tr_idx[a], tr_idx[b]
        _, par = izgara_sec(ad, X_tem, SAY_F, SAP_F, a_idx, MASKE, cv=ic)
        e = a_idx[~MASKE[a_idx]]
        m = model_kur(ad, SAY_F, SAP_F, **par).fit(X_tem.loc[e], y[e])
        dis_skor.append(average_precision_score(y[b_idx], m.predict_proba(X_tem.loc[b_idx])[:, 1]))
        secilen.append(par)
    d = np.array(dis_skor)
    duz = skor[(skor.Asama.str.startswith("4")) & (skor.Model == ad)]["CV PR-AUC ort"].iloc[0]
    R["nested"][ad] = {"ort": float(d.mean()), "sapma": float(d.std()), "duz_cv": float(duz),
                       "secilenler": [str(s) for s in secilen]}
    print(f"  {ad:<20} nested {d.mean():.4f} +- {d.std():.4f} | duz CV {duz:.4f} | "
          f"dis katlarda secilen: {secilen}")

# =============================================================================
# SORU 2c - En yuksek accuracy en iyi model mi?
# =============================================================================
baslik("SORU 2c - Accuracy siralamasi vs digerleri (asama 4)")
s4 = skor[skor.Asama.str.startswith("4")].set_index("Model")
for met in ["Accuracy", "Precision", "Recall", "F1", "PR-AUC"]:
    print(f"  {met:<10}: " + " > ".join(f"{m} ({s4.loc[m, met]:.4f})"
                                         for m in s4[met].sort_values(ascending=False).index))
R["2c"] = {met: s4[met].sort_values(ascending=False).round(4).to_dict()
           for met in ["Accuracy", "Precision", "Recall", "F1", "PR-AUC"]}

# =============================================================================
# SORU 2d - Model secimi: yorumlanabilir kurallar
# =============================================================================
baslik("SORU 2d - Secilen model: Karar Agaci, kurallar ve degisken onemi")
SECILEN = "Decision Tree"
agac = MODEL_F[SECILEN].named_steps["clf"]
ADLAR = SAY_F + ([GucSapmasi.AD] if SAP_F else []) + ["Type"]
kurallar = export_text(agac, feature_names=ADLAR, max_depth=3, decimals=1)
print(kurallar)
(OUT / "s2d_agac_kurallari.txt").write_text(export_text(agac, feature_names=ADLAR, decimals=1), encoding="utf-8")
R["2d"] = {"yaprak": int(agac.get_n_leaves()), "derinlik": int(agac.get_depth()), "kurallar_3": kurallar}

perm = permutation_importance(MODEL_F[SECILEN], X_tem.loc[te_idx, SUT_F], y[te_idx], n_repeats=10,
                              random_state=SEED, scoring="average_precision")
onem = pd.DataFrame({"Degisken": SUT_F, "Permutasyon onemi": perm.importances_mean})
gini = dict(zip(ADLAR, agac.feature_importances_))
onem["Gini onemi"] = onem["Degisken"].map(gini)
if SAP_F:   # Pipeline icinde uretilen sutun: permutasyonu tork/devir uzerinden dolayli olculur
    onem = pd.concat([onem, pd.DataFrame([{"Degisken": GucSapmasi.AD, "Permutasyon onemi": np.nan,
                                           "Gini onemi": gini[GucSapmasi.AD]}])], ignore_index=True)
onem = onem.sort_values("Gini onemi", ascending=False)
onem["Turetilmis"] = ~onem["Degisken"].isin(HAM)
onem.round(4).to_csv(OUT / "s2d_degisken_onemi.csv", index=False)
print(onem.round(4).to_string(index=False))
R["2d"]["onem"] = onem.round(4).to_dict("records")
R["2d"]["turetilmis_gini_pay"] = float(onem.loc[onem.Turetilmis, "Gini onemi"].sum())

fig, ax = plt.subplots(figsize=(13, 4.6))
plot_tree(agac, max_depth=3, feature_names=ADLAR, class_names=["saglam", "arizali"], filled=True,
          impurity=False, proportion=False, fontsize=6, ax=ax)
plt.tight_layout(); plt.savefig(GRAF / "soru2_karar_agaci.png", dpi=170); plt.close()

# =============================================================================
# SORU 3a - Hangi hata daha kritik? Maliyet tabanli esik (OOF'tan, test acilmadan)
# =============================================================================
baslik("SORU 3a - Maliyet orani -> esik (OOF), sonra testte bir kez")
yd, p_oof = y[tr_idx], OOF_F[SECILEN]
adaylar = np.unique(np.round(p_oof, 4))


def maliyet_esigi(oran):
    en = None
    for t in adaylar:
        tp, fp, fn, tn = karmasa(yd, p_oof, t)
        c = oran * fn + fp
        if en is None or c < en[0]:
            en = (c, t)
    return float(en[1])


mal = []
for oran in [1, 5, 10, 20, 50]:
    t = maliyet_esigi(oran)
    tp, fp, fn, tn = karmasa(y[te_idx], TEST_P[SECILEN], t)
    otp, ofp, ofn, otn = karmasa(yd, p_oof, t)
    mal.append({"Oran (1 FN = k FP)": oran, "Esik": t,
                "OOF Recall": otp / (otp + ofn), "OOF FP": ofp,
                "Test TP": tp, "Test FP": fp, "Test FN": fn,
                "Test Recall": tp / (tp + fn), "Test Precision": tp / (tp + fp) if tp + fp else 0.0,
                "Test maliyet": oran * fn + fp})
mal = pd.DataFrame(mal)
mal.round(4).to_csv(OUT / "s3a_maliyet_esik.csv", index=False)
print(mal.round(4).to_string(index=False))
T_NIHAI = float(mal.loc[mal["Oran (1 FN = k FP)"] == MALIYET_ORANI, "Esik"].iloc[0])

# esigin katlar arasi kararliligi (yalnizca OOF; test kullanilmaz)
kat_recall = []
for a, b in SKF.split(X_tem.loc[tr_idx], yd):
    tp, fp, fn, tn = karmasa(yd[b], p_oof[b], T_NIHAI)
    kat_recall.append(tp / (tp + fn))
print(f"Nihai esik {T_NIHAI} | OOF katlarda recall: {np.round(kat_recall, 3)}")

tp, fp, fn, tn = karmasa(y[te_idx], TEST_P[SECILEN], T_NIHAI)
nihai = {"esik": T_NIHAI, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
         "Recall": tp / (tp + fn), "Precision": tp / (tp + fp), "F1": 2 * tp / (2 * tp + fp + fn),
         "Accuracy": (tp + tn) / len(te_idx),
         "PR-AUC": float(average_precision_score(y[te_idx], TEST_P[SECILEN])),
         "ROC-AUC": float(roc_auc_score(y[te_idx], TEST_P[SECILEN])),
         "kat_recall": [float(v) for v in kat_recall]}
print("Nihai (test):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in nihai.items()})

# Politika esikleri (uc model, OOF'tan secilir)
POL = ["Ariza kacirma (recall>=0,80)", "Yanlis alarm sinirla (FPR<=0,10)", "En yuksek F1"]
pol_satir = []
for ad in MODELLER:
    p = OOF_F[ad]; ad_t = np.unique(np.round(p, 4))
    pr, rc, th = precision_recall_curve(yd, p)
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
    uy = [t for t in ad_t if (p[yd == 1] >= t).mean() >= .80]
    uy2 = [t for t in ad_t if (p[yd == 0] >= t).mean() <= .10]
    es = {POL[0]: max(uy) if uy else 0.0, POL[1]: min(uy2) if uy2 else 1.0,
          POL[2]: th[int(np.argmax(f1[:-1]))]}
    for pol, t in es.items():
        tp_, fp_, fn_, tn_ = karmasa(y[te_idx], TEST_P[ad], t)
        pol_satir.append({"Politika": pol, "Model": ad, "Esik": float(t), "TP": tp_, "FP": fp_,
                          "FN": fn_, "TN": tn_, "Recall": tp_ / (tp_ + fn_),
                          "Precision": tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0})
pol = pd.DataFrame(pol_satir)
pol.round(4).to_csv(OUT / "s3a_politika_esikleri.csv", index=False)
print(pol.round(4).to_string(index=False))
R["3a"] = {"maliyet": mal.round(4).to_dict("records"), "nihai": nihai,
           "politika": pol.round(4).to_dict("records"), "oran": MALIYET_ORANI}

# --- test belirsizligi: bootstrap %95 guven araliklari
rng = np.random.default_rng(SEED)
yt, pt, pk = y[te_idx], TEST_P[SECILEN], TEST_P["KNN"]
bs = {"Recall": [], "Precision": [], "PR-AUC": [], "PR-AUC farki (DT-KNN)": []}
for _ in range(2000):
    s = rng.integers(0, len(yt), len(yt))
    if yt[s].sum() == 0:
        continue
    yp = (pt[s] >= T_NIHAI).astype(int)
    bs["Recall"].append(recall_score(yt[s], yp))
    bs["Precision"].append(precision_score(yt[s], yp, zero_division=0))
    bs["PR-AUC"].append(average_precision_score(yt[s], pt[s]))
    bs["PR-AUC farki (DT-KNN)"].append(bs["PR-AUC"][-1] - average_precision_score(yt[s], pk[s]))
R["bootstrap"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
print("Bootstrap %95 GA:", {k: np.round(v, 3).tolist() for k, v in R["bootstrap"].items()})

# =============================================================================
# SORU 3b - Arizalari yakalama basarisini artirma
# =============================================================================
baslik("SORU 3b - Recall artirma denemeleri (OOF, sonra test)")
den = []


def dene(ad, mdl, t=0.5):
    cvs, oof = cv_skor(mdl, X_tem, tr_idx, MASKE, oof=True)
    otp, ofp, ofn, _ = karmasa(yd, oof, t)
    _, p, _ = test_olc(mdl, X_tem, EGIT)
    ttp, tfp, tfn, _ = karmasa(y[te_idx], p, t)
    den.append({"Deneme": ad, "CV PR-AUC": cvs.mean(), "OOF Recall": otp / (otp + ofn),
                "OOF Precision": otp / (otp + ofp) if otp + ofp else 0.0,
                "Test Recall": ttp / (ttp + tfn), "Test FP": tfp, "Test FN": tfn})


A = AYAR[SECILEN]
dene("A) Ayarli agac, esik 0,50", model_kur(SECILEN, SAY_F, SAP_F, **A))
dene(f"B) A + esik {T_NIHAI} (maliyet 1:{MALIYET_ORANI})", model_kur(SECILEN, SAY_F, SAP_F, **A), T_NIHAI)
dene("C) A + class_weight=balanced", model_kur(SECILEN, SAY_F, SAP_F, class_weight="balanced", **A))
t20 = float(mal.loc[mal["Oran (1 FN = k FP)"] == 20, "Esik"].iloc[0])
dene(f"D) A + esik {t20} (maliyet 1:20)", model_kur(SECILEN, SAY_F, SAP_F, **A), t20)
den = pd.DataFrame(den)
den.round(4).to_csv(OUT / "s3b_recall_denemeleri.csv", index=False)
print(den.round(4).to_string(index=False))

kac = df.loc[te_idx][(y[te_idx] == 1) & (TEST_P[SECILEN] < T_NIHAI)]
tur = pd.DataFrame({"Toplam": df.loc[te_idx][y[te_idx] == 1][YASAK].sum(), "Kacirilan": kac[YASAK].sum()})
tur["Yakalama %"] = (100 * (1 - tur["Kacirilan"] / tur["Toplam"].replace(0, np.nan))).round(1)
print("\nAriza turune gore yakalama (YASAK sutunlar yalnizca hata analizinde, girdi degil):")
print(tur.to_string())
R["3b"] = {"denemeler": den.round(4).to_dict("records"),
           "tur": {k: {kk: (None if pd.isna(vv) else float(vv)) for kk, vv in v.items()}
                   for k, v in tur.to_dict("index").items()}}

# =============================================================================
# SORU 3c - Egitimde yuksek, testte dusuk: asiri ogrenme teshisi
# =============================================================================
baslik("SORU 3c - Egitim vs test, derinlik egrisi")
ezb = []
for ad in MODELLER:
    for etiket, ayar in [("varsayilan", {}), ("ayarli", AYAR[ad])]:
        m = model_kur(ad, SAY_F, SAP_F, **ayar).fit(X_tem.loc[EGIT], y[EGIT])
        ezb.append({"Model": ad, "Surum": etiket,
                    "Egitim PR-AUC": average_precision_score(y[EGIT], m.predict_proba(X_tem.loc[EGIT])[:, 1]),
                    "Test PR-AUC": average_precision_score(y[te_idx], m.predict_proba(X_tem.loc[te_idx])[:, 1])})
ezb = pd.DataFrame(ezb); ezb["Fark"] = ezb["Egitim PR-AUC"] - ezb["Test PR-AUC"]
ezb.round(4).to_csv(OUT / "s3c_egitim_test.csv", index=False)
print(ezb.round(4).to_string(index=False))

der = []
for d in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None]:
    mdl = model_kur(SECILEN, SAY_F, SAP_F, max_depth=d)
    cvs = cv_skor(mdl, X_tem, tr_idx, MASKE)
    m = clone(mdl).fit(X_tem.loc[EGIT], y[EGIT])
    der.append({"max_depth": "sinirsiz" if d is None else d,
                "Egitim PR-AUC": average_precision_score(y[EGIT], m.predict_proba(X_tem.loc[EGIT])[:, 1]),
                "CV PR-AUC": cvs.mean(), "CV sapma": cvs.std(),
                "Yaprak": int(m.named_steps["clf"].get_n_leaves())})
der = pd.DataFrame(der)
der.round(4).to_csv(OUT / "s3c_derinlik_egrisi.csv", index=False)
print(der.round(4).to_string(index=False))
R["3c"] = {"egitim_test": ezb.round(4).to_dict("records"), "derinlik": der.round(4).to_dict("records")}

fig, ax = plt.subplots(1, 2, figsize=(10, 2.8))
xl = range(len(der))
ax[0].plot(xl, der["Egitim PR-AUC"], "o-", c="#E8684A", label="egitim")
ax[0].errorbar(xl, der["CV PR-AUC"], yerr=der["CV sapma"], fmt="s-", c="#5B8FF9", capsize=3, label="capraz dogrulama")
ax[0].set_xticks(list(xl)); ax[0].set_xticklabels(der["max_depth"], fontsize=7)
ax[0].set_xlabel("max_depth"); ax[0].set_ylabel("PR-AUC"); ax[0].legend(fontsize=7)
ax[0].set_title("3c) Derinlik arttikca ezber: egitim 1,0'a, CV asagi")
ax[1].plot(mal["Oran (1 FN = k FP)"].astype(str), mal["Test Recall"], "o-", c="#E8684A", label="recall")
ax[1].plot(mal["Oran (1 FN = k FP)"].astype(str), mal["Test Precision"], "s-", c="#5B8FF9", label="precision")
ax[1].set_xlabel("1 kacirilan ariza = k yanlis alarm"); ax[1].legend(fontsize=7)
ax[1].set_title("3a-3b) Maliyet orani buyudukce esik duser")
plt.tight_layout(); plt.savefig(GRAF / "soru3.png", dpi=170); plt.close()

# =============================================================================
# SIZINTI DENETIMI ve TESLIM DOSYALARI
# =============================================================================
baslik("Sizinti denetimi")
kontrol = {
    "Yasak sutunlar (TWF/HDF/PWF/OSF/RNF) girdide yok": not any(c in SUT_F for c in YASAK),
    "UDI / Product ID girdide yok": not any(c in SUT_F for c in KIMLIK),
    "Egitim ve test kesisimi bos": len(set(tr_idx) & set(te_idx)) == 0,
    "Her UDI tek satir (grup sizintisi yok)": df["UDI"].is_unique,
    "Test kumesi butun asamalarda ayni (2000 satir, hic satir atilmadi)": len(te_idx) == 2000,
    "Nihai egitim satirlari test ile kesismiyor": len(set(EGIT) & set(te_idx)) == 0,
    "En guclu tekli degisken ROC-AUC < 0,80": float(tekli["ROC-AUC"].max()) < 0.80,
}
for k, v in kontrol.items():
    print(f"  [{'OK' if v else 'DIKKAT'}] {k}")
R["sizinti"] = {k: bool(v) for k, v in kontrol.items()}


def temiz_ad(c):
    for k in "[]<>:":
        c = c.replace(k, "")
    return c.strip().replace(" ", "_")


Xteslim = X_tem[SUT_F].copy()
if SAP_F:
    Xteslim = GucSapmasi().fit(X_tem.loc[EGIT]).transform(Xteslim)
Xteslim.columns = [temiz_ad(c) for c in Xteslim.columns]
Xteslim.insert(0, "UDI", df["UDI"].values)
Xteslim[TARGET] = y
Xteslim.loc[EGIT].to_csv(OUT / "islenmis_egitim.csv", index=False)
Xteslim.loc[te_idx].to_csv(OUT / "islenmis_test.csv", index=False)

import sklearn, scipy, matplotlib as mpl
R["surumler"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                 "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "matplotlib": mpl.__version__}
with open(OUT / "sonuclar.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("\nBitti. Ciktilar:", OUT)
