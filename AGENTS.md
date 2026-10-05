# AGENTS.md — Equipo DataFest (reto BCP × ESAN)

Este archivo es el cerebro del equipo. Lo carga cualquier agente (Claude Code, OpenCode,
Copilot, Cursor) antes de tocar el reto. Junta tres cosas:

1. **Lo que aprendimos en retos pasados** (DataFest 2025 y este ensayo 2026).
2. **Lo que sabemos de banca**: propensión, campañas, canales, riesgo y valor del cliente.
3. **Cómo trabajamos**: roles, orden de los pasos, qué se entrega y cómo se vende.

La meta: que el día del reto final el trabajo pesado ya esté resuelto y solo pidamos
afinamientos del tipo *"prueba CatBoost"*, *"añade estas variables"* o *"sube la capacidad
del call center a 3.000"*.

---

## 0. Antes de cualquier cosa (la hora cero)

1. Leer el enunciado completo y llenar [`RETO.md`](RETO.md) con la **ficha del reto**:
   objetivo, métrica, unidad de análisis, columnas, formato de entrega, restricciones y
   fecha/hora de entrega. Si algo no está claro, se anota como **supuesto** con su porqué.
2. Ajustar [`src/config.py`](src/config.py): rutas, columna objetivo, id, columna de tiempo,
   métrica y formato de entrega. **Todo el código lee de ahí**; nada se escribe a mano en los
   scripts.
3. Correr `python -m src.de_profile` y leer `reports/perfil.md` antes de
   proponer cualquier modelo.
4. Hacer una **entrega de seguridad** en la primera hora (línea base simple, formato
   validado). Desde ahí solo se mejora.

## 1. Reglas de oro (no se negocian)

| # | Regla | Por qué (lo aprendimos así) |
|---|---|---|
| R1 | **Si la métrica es de ordenamiento (AUC/Gini), se entrega la probabilidad, nunca 0/1.** | En 2025 binarizar dejó el Gini en ~0,22. Si el reto exige 0/1, el umbral se elige maximizando la métrica oficial en validación, no F1 ni 0,5. |
| R2 | **Validación temporal, nunca aleatoria.** Entrenar con el pasado y validar con el mes siguiente (walk-forward). | El test siempre es el mes más reciente. Un KFold aleatorio filtra el futuro y promete un Gini que no existe. |
| R3 | **La validación debe parecerse al test.** Misma composición (clientes que siguen + clientes nuevos), mismo horizonte. | En el ensayo 2026 el test es: no convertidos del último mes + clientes nuevos. |
| R4 | **Cobertura del 100 %.** Una predicción por fila del test, en el orden original, sin nulos, dentro de [0, 1]. | Las métricas de 2025 penalizaban la cobertura. `src/submit.py` lo valida y corta si falla. |
| R5 | **Cero fuga de información.** Ninguna variable puede usar datos del mismo mes del objetivo ni del futuro. Las variables de historia se calculan con `shift`. | Es el error que más Gini falso produce en datos de panel. |
| R6 | **Desconfiar de lo perfecto.** Tasas de 0 % o 100 %, una variable con Gini > 0,6 sola, columnas duplicadas: se investigan antes de usarlas. | En 2025, `prestamo` y `vehicular` tenían 100 % de venta: era un sesgo de selección, no señal. |
| R7 | **Optimización exacta, no por orden de prioridad.** Si hay restricciones, se formula como programación lineal entera (OR-Tools) y se compara contra la versión greedy. | En 2025 quedó pendiente: el greedy deja valor sobre la mesa. |
| R8 | **Probabilidades calibradas cuando alimentan dinero.** Para el ranking no importa; para el valor esperado (p × monto) sí. | Sin calibrar, el valor esperado en soles queda inflado o desinflado. |
| R9 | **Cada experimento queda registrado** en `reports/experimentos.csv` (fecha, quién, cambio, Gini por fold, promedio, desviación). | Sin registro no se puede defender qué funcionó ni volver atrás. |
| R10 | **Todo número de la presentación sale de un archivo del repo.** | El jurado pregunta. Hay que poder abrir el archivo y mostrarlo. |
| R11 | **Caja fuerte.** El último mes de train (`config.HOLDOUT_T`) no se usa para decidir nada durante la búsqueda. Solo se abre con `--holdout` para confirmar como máximo 3 candidatos por ronda. | Con señal baja, iterar cientos de veces contra la misma validación fabrica mejoras falsas. En el ensayo, los folds daban 0,259 y la caja fuerte 0,230. |
| R13 | **Con señal baja, variables antes que tuning, y el campeón lo elige la caja fuerte.** Si la brecha logística-boosting es < 0,03, primero se buscan reglas con umbral en cruces de 2 o 3 variables y se prueban en una logística. Ante empate en los folds, gana el modelo más simple. | Ensayo, ronda 1: las 4 reglas de DE subieron la logística +0,027; Optuna subió LightGBM +0,004. En la caja fuerte ganó la logística sola (0,247) frente al ensamble (0,240) y a LightGBM afinado (0,235). |
| R14 | **Ningún comando largo sin señales de vida.** Más de ~8 min sin salida, o un estudio de Optuna sin storage persistente, no se lanza: se corre en tandas que se retoman (`reports/tuning/*.db`). | La primera corrida de Optuna murió en la prueba 19 y se perdió entera; los agentes se cortaron por 10 min sin progreso. |
| R12 | **Notebook para explorar, `src/` para producir.** Toda exploración vive en `notebooks/NN_tema.ipynb`, ejecutado y con sus salidas. Lo que funciona se mueve a `src/` y el notebook lo importa desde ahí. | El notebook cuenta la historia y se muestra al jurado; `src/` garantiza que la entrega use el mismo código. |

## 2. El flujo y los roles

```
Ficha del reto ─► Data Engineering ─► ML ─► Optimización ─► Caso de negocio ─► Entrega + pitch
  (todos)          (perfil, EDA,       (modelos,  (restricciones, (lift, soles,     (CSV validado,
                    limpieza,           validación, asignación)     historia)         presentación)
                    variables)          ensamble)
```

El **caso de negocio** no es un paso al final: cada rol deja en su reporte una sección
*"¿Qué significa esto para el banco?"*.

| Rol | Playbook | Entrega |
|---|---|---|
| Data Engineer | [`playbooks/01-data-engineering.md`](playbooks/01-data-engineering.md) | `reports/perfil.md`, `reports/eda.md`, `data/interim/base.parquet`, `data/interim/features.parquet` |
| ML | [`playbooks/02-ml.md`](playbooks/02-ml.md) | `reports/experimentos.csv`, `models/`, predicciones out-of-fold, `reports/ml.md` |
| Optimización | [`playbooks/03-optimizacion.md`](playbooks/03-optimizacion.md) | `reports/optimizacion.md`, `outputs/asignacion.csv` |
| Negocio (todos, con un dueño) | [`playbooks/00-caso-de-negocio.md`](playbooks/00-caso-de-negocio.md) | `reports/negocio.md`, cifras de la presentación |
| Entrega | [`playbooks/04-entrega-y-pitch.md`](playbooks/04-entrega-y-pitch.md) | `outputs/submission.csv`, presentación |

### Cada rol es un agente, y los agentes se hablan por la bitácora
Cada rol puede correr como un agente independiente. Los agentes **no comparten memoria**:
solo saben lo que el otro dejó escrito en `comms/bitacora.jsonl` y en los archivos del repo.
Protocolo completo en [`playbooks/comunicacion.md`](playbooks/comunicacion.md). En resumen:

- **Al empezar**: `python -m src.comms leer --para <rol>` y responder los pedidos pendientes.
- **Durante**: un `hallazgo` cuando algo cambia la estrategia de otro rol, un `pedido` cuando
  se necesita algo de otro y un `bloqueo` cuando no se puede seguir.
- **Al terminar**: un `handoff` con qué se entrega, dónde y con qué experimento, y una o
  más `leccion` con lo que debería pasar al AGENTS.md.
- `python -m src.comms visor` genera `comms/visor.html` para revisar y depurar la conversación.

### El ciclo de mejora (cómo itera un agente solo)
1. **Hipótesis** escrita antes de correr ("X debería subir el Gini porque...").
2. **Experimento** con `src.train` (queda en `experimentos.csv`, regla R9).
3. **Decisión** con la regla de adopción de §3. Si no mejora, se descarta y se anota el porqué.
4. **Lección**: lo aprendido se envía como `leccion` y el coordinador lo sube al AGENTS.md o al playbook.
5. Cada ronda cierra con un **campeón** confirmado en la caja fuerte (R11).

Conocimiento de apoyo:
- [`playbooks/lecciones-retos-pasados.md`](playbooks/lecciones-retos-pasados.md): qué funcionó y qué falló en 2025 y en el ensayo.
- [`playbooks/banca.md`](playbooks/banca.md): glosario y lógica de negocio de banca retail en Perú.

## 3. Cómo pedirle cosas al agente el día del reto

El agente debe entender estos pedidos cortos y ejecutarlos de punta a punta: código,
validación temporal, registro en `experimentos.csv` y una línea de resultado.

| Pedido | Qué hace el agente |
|---|---|
| `perfila el dataset` | Corre `src.de_profile` y resume los 5 hallazgos más importantes y los riesgos de fuga. |
| `añade la variable X` | La agrega en `src/features.py` (con `shift` si usa historia), re-entrena el modelo campeón y reporta el delta de Gini por fold. |
| `prueba el modelo X` | Lo agrega a `src/models.py` con la misma interfaz, corre la validación temporal y lo compara con el campeón. |
| `ensambla` | Combina los modelos registrados por promedio de rankings y busca pesos con las predicciones out-of-fold. |
| `optimiza con capacidad N` | Cambia `config.OPT` y vuelve a resolver; reporta el valor esperado contra greedy y contra aleatorio. |
| `genera la entrega` | Re-entrena con todo el train, predice el test, valida el formato (R4) y escribe `outputs/submission.csv`. |
| `actualiza el caso de negocio` | Recalcula lift, captura por decil y soles en `reports/negocio.md`. |

Siempre se reporta: **Gini medio ± desviación por fold**, el delta contra el campeón y
si se adopta o no. Un cambio se adopta si mejora el promedio **y** no empeora más de un
fold de forma clara.

## 4. Convenciones del código

- Python 3.9+, `pandas`, `lightgbm`, `catboost`, `xgboost`, `optuna`, `shap`, `ortools`.
  Entorno: `.venv/` (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`).
- Estructura:
  ```
  src/config.py        todo lo específico del reto (rutas, columnas, métrica, supuestos de negocio)
  src/de_profile.py    análisis preliminar -> reports/perfil.md
  src/de_clean.py      tipos, limpieza, chequeos -> data/interim/base.parquet
  src/features.py      variables (historia con shift) -> data/interim/features.parquet
  src/validation.py    folds temporales y métrica oficial
  src/models.py        modelos con interfaz fit/predict_proba común
  src/train.py         corre experimentos, registra, guarda OOF
  src/optimize.py      asignación con restricciones (OR-Tools)
  src/business.py      lift, deciles, valor esperado
  src/submit.py        entrega final + validación de formato
  src/comms.py         bitácora entre agentes + visor
  notebooks/           exploración ejecutada (R12): perfil, EDA, laboratorio de variables, ML
  comms/               bitacora.jsonl y visor.html
  ```
- Varios agentes corren en paralelo en la misma máquina: cada proceso usa `config.N_JOBS`
  hilos. No se editan archivos de otro rol sin pedirlo por la bitácora.
- Todo script se corre como módulo desde la raíz: `python -m src.train --modelo lgbm`.
- Semilla fija (`config.SEED`). Nada aleatorio sin semilla.
- Los archivos de datos originales **no se modifican**. Lo derivado va a `data/interim/`.
- Los reportes en `reports/` se escriben en español, con tablas cortas y una sección de
  negocio al final.

## 5. Cómo se vende (lo que gana el reto)

El jurado del BCP y de ESAN no premia el modelo más complejo, sino **la decisión mejor
defendida**. Toda historia sigue este orden:

1. **El problema en soles**: cuánto cuesta contactar a ciegas.
2. **Qué descubrimos** en los datos: 3 hallazgos, cada uno con su gráfico.
3. **Cómo lo resolvimos**: modelo + optimización, explicado sin jerga.
4. **Cuánto vale**: lift, conversiones capturadas, soles por campaña, contra no hacer nada.
5. **Cómo se pone en producción**: reentrenamiento mensual, monitoreo y riesgos.

Ver [`playbooks/04-entrega-y-pitch.md`](playbooks/04-entrega-y-pitch.md).
