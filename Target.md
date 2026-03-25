# 🧠 Desktop + Browser Agent Architecture

## 1) Capa de percepción

El agente no debe ver solo “una web” o solo “una captura”, sino un entorno mixto.

Debe poder recibir:

- DOM y elementos interactivos del navegador
- Screenshots del navegador
- Screenshots del escritorio completo
- OCR del texto visible
- Datos del sistema:
  - ventana activa
  - app activa
  - foco de teclado

Esto le permite decidir si una tarea va por navegador, por escritorio, o por ambos.

---

## 2) Capa de control

Separar dos controladores:

### Browser Controller
- Navegación web
- Formularios
- Filtros
- Búsquedas
- Descargas
- Logins

### Desktop Controller
- Ratón
- Teclado
- Atajos
- Ventanas
- Apps
- IDEs
- Terminal
- Diálogos del sistema

👉 El agente central no hace clicks ni escribe directamente; solo decide.  
👉 Los controladores ejecutan.

---

## 3) Capa de planificación

Mantener la idea de **Planner + Executor**.

- El **Planner** decide la estrategia general.
- El **Executor** hace pasos pequeños y verificables.
- Si algo falla, el Planner replanifica.

👉 Esto evita que el modelo pequeño tenga que resolver todo de golpe.

---

# 🧠 Memoria persistente (núcleo del aprendizaje)

No plantearlo como entrenamiento de pesos al inicio, sino como:

👉 **Memoria estructurada que se enriquece con el uso**

---

## Tipos de memoria

### A) Memoria episódica

Guarda sesiones reales:

- tarea objetivo
- sitio / app
- pasos realizados
- capturas
- errores
- resultado final

👉 Sirve para recordar “cómo se resolvió algo parecido”.

---

### B) Memoria semántica

Guarda patrones abstractos:

- “en este portal, el filtro remoto está en tal zona”
- “esta app suele abrir con este popup”
- “este tipo de formulario usa Enter y no botón”

👉 Sirve para generalizar.

---

### C) Memoria procedimental

Guarda habilidades reutilizables:

- buscar empleo en portal X
- aplicar filtro remoto
- abrir proyecto en Cursor
- crear branch, editar archivo, correr tests

👉 Son tus **skills/macros**

---

# 🔁 Entrenamiento real (sin fine-tuning inicial)

## Fase 1: Observación

Tras cada tarea:

- guardar contexto
- guardar acciones
- guardar qué funcionó
- guardar qué falló
- guardar capturas antes/después

---

## Fase 2: Clasificación

Etiquetar la experiencia:

- portal de empleo
- login wall
- cookie banner
- búsqueda
- filtro
- scroll infinito
- formulario
- IDE
- terminal
- error de red

---

## Fase 3: Abstracción

Extraer patrones:

- “el buscador está en header”
- “el filtro remoto abre panel lateral”

---

## Fase 4: Promoción a skill

Cuando un patrón funciona repetidamente:

👉 se convierte en una **skill reutilizable**

---

## Fase 5: Recuperación en runtime

Para nuevas tareas:

- buscar casos similares
- cargar skill probable
- adaptarla con el Planner

👉 Esto crea la sensación de aprendizaje.

---

# 🧩 Modelo de datos (memoria)

Objetos clave:

- **Task** → objetivo humano
- **Session** → ejecución concreta
- **Trace** → secuencia de acciones/observaciones
- **Skill** → patrón reutilizable
- **SiteProfile** → perfil de portal/app
- **UIAtom** → elemento típico de interfaz
- **FailurePattern**
- **RecoveryPattern**

---

## Ejemplo de UI Atoms (portal de empleo)

- buscador principal
- filtro ubicación
- toggle remoto
- selector de fecha
- botón aplicar
- login modal
- cookie banner
- listado de resultados

👉 Evita redescubrir la UI cada vez.

---

# 🧠 Motor de clasificación

No resuelve tareas. Clasifica:

- tipo de tarea
- tipo de entorno
- UI presente
- skill reutilizable
- nivel de riesgo
- modalidad (browser / desktop / híbrido)

### Ejemplo


task_family = job_search
environment = browser
site_profile = portal_x
ui_atoms = search, filter, apply_button
skill_candidate = job_search_portal_x_v2


---

# 💾 Persistencia de memoria

## 1) Memoria operativa
- contexto inmediato
- sesiones recientes

## 2) Memoria larga

- base relacional (estructura)
- buscador semántico (similitud)
- almacenamiento de capturas + metadata

### Debe permitir buscar por:

- sitio
- tipo de tarea
- éxito/fallo
- similitud visual
- similitud textual

---

# 🖥️ Desktop + Browser Controller

## Modo navegador

- buscar
- formularios
- scraping
- navegación

## Modo escritorio

- abrir apps
- terminal
- mover ventanas
- IDEs
- PDFs
- sistema

## Modo híbrido

- navegador → docs
- escritorio → código
- terminal → tests
- navegador → validación

👉 Esto es un **asistente real**

---

# 👨‍💻 Modo desarrollo (Cursor / Antigravity)

## Project Memory

- estructura del repo
- propósito
- archivos clave
- decisiones de arquitectura
- comandos útiles
- tests
- errores frecuentes
- checklist release

---

## Acciones del agente

- leer archivo
- buscar símbolo
- abrir terminal
- ejecutar tests
- revisar diff
- proponer cambios
- confirmar acciones destructivas

---

# 🗺️ Feature Plan

## Fase 1 — Base sólida
- estabilizar loop
- separar observación/acción
- logging completo
- capturas + DOM
- fallback

---

## Fase 2 — Memoria persistente
- memoria episódica
- memoria semántica
- skills reutilizables
- perfiles por sitio
- búsqueda por similitud

---

## Fase 3 — Clasificación
- task family
- UI atoms
- popups / login walls
- tracking éxito/fallo

---

## Fase 4 — Browser Controller
- navegación
- formularios
- filtros
- extracción
- recovery
- replanificación

---

## Fase 5 — Desktop Controller
- mouse / teclado
- screenshots
- OCR
- ventanas
- foco
- atajos

---

## Fase 6 — Orquestación híbrida
- router browser/desktop
- Planner decide
- Executor ejecuta
- recovery automático

---

## Fase 7 — Modo desarrollo
- memoria de proyecto
- repo awareness
- edición
- terminal
- tests
- revisión

---

## Fase 8 — Mejora continua
- ranking de skills
- detección de fallos
- mejora de prompts
- fine-tuning futuro

---

# 🧭 Estrategia recomendada

1. Memoria + clasificación
2. Browser controller
3. Desktop controller
4. Modo híbrido
5. Modo desarrollo

👉 No al revés (dificultad explota)

---

# 🚫 Evitar

- entrenar modelo desde el inicio
- una sola llamada LLM para todo
- logs sin estructura
- mezclar todo desde el día 1
- usar screenshots sin OCR

---

# 🧠 Arquitectura ideal (resumen)

## Input
- DOM
- Screenshot
- OCR
- Contexto sistema

## Core
- clasificador
- planner
- memoria de skills
- retriever

## Execution
- browser controller
- desktop controller

## Learning
- guardar trazas
- extraer patrones
- promover skills
- reutilizar