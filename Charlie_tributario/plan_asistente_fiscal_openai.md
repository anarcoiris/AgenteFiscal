# Plan completo para integrar GPT‑4.1 en un asistente fiscal con captura de pantalla

## 1. Requisitos funcionales y modos de captura de pantalla

- **Captura de pantalla**: Debe permitir al usuario hacer capturas del contenido relevante. Se puede usar PyAutoGUI en Python, cuya función `screenshot()` devuelve la imagen de la pantalla (o de una región)【24†L52-L58】. PyAutoGUI soporta el parámetro `region=(left, top, width, height)` para capturar solo un área específica【24†L52-L58】. En sistemas operativos también se pueden invocar comandos nativos (`screencapture` en macOS, `scrot` en Linux, etc.).
- **Selección de área interactiva**: El usuario debe poder definir qué parte de la pantalla analizar. Una opción es crear una ventana GUI transparente (p.ej. con **PySimpleGUI**【10†L65-L74】) que el usuario arrastre para delimitar el área de interés. El programa oculta esa ventana antes de tomar la captura para no bloquear la vista【19†L53-L59】. Por ejemplo, un tutorial muestra cómo usar PySimpleGUI para dibujar una ventana semitransparente y luego tomar screenshot de su posición【10†L65-L74】.
- **Extensión o webapp**: Como alternativa a la app local, se puede desarrollar una extensión de navegador o webapp popup. Por ejemplo, la extensión **ContextCapture** para Chrome/Edge permite seleccionar un área de la página con un overlay interactivo, hace OCR y luego genera un resumen con GPT【8†L293-L301】【8†L358-L366】. Esto demuestra que una extensión puede implementar la lógica: clic en icono, arrastrar región, extraer texto y consultar al LLM.
- **Modos de captura**:
  - *Manual*: Botón o atajo que inicia la captura cuando el usuario lo desee.
  - *Automático*: Capturas periódicas cada X segundos/minutos ajustables, para monitorizar cambios (por ejemplo, analizar la pantalla cada cierto tiempo).
- **Opciones de selección**: Debe haber al menos dos modos: 
  - *Pantalla completa*: captura toda la pantalla actual.
  - *Ventana o región específica*: permitir escoger una ventana abierta (por título/ID) o dibujar un rectángulo en la pantalla.
- **Configuración ajustable**: Parámetros como la escala de la captura (p.ej. en pantallas Retina), la resolución o calidad de imagen y el idioma de OCR (por defecto español) deberían ser configurables.
- **Procesamiento inicial**: La aplicación debe detectar texto en la imagen (usar OCR). Se busca texto relacionado con temas fiscales (p.ej. palabras clave como *IRPF*, *IVA*, *rendimientos*, *base imponible*). Este texto extraído será la base para el siguiente paso de RAG (ver más abajo).  

  

## 2. APIs de OpenAI y límites de GPT‑4.1 para RAG

- **Disponibilidad de GPT-4.1**: GPT-4.1 solo está disponible a través de la API de OpenAI (no en la interfaz pública de ChatGPT)【1†L99-L104】. Por tanto, la integración del modelo se hará mediante llamadas a la API (Chat Completions o Responses API).
- **Ventana de contexto**: GPT-4.1 admite hasta ~1.000.000 tokens de contexto【1†L50-L57】【32†L158-L160】. Este enorme contexto permite procesar documentos extensos de una sola vez. (Sin embargo, en contextos muy largos la precisión puede degradarse ligeramente【32†L158-L160】, por lo que suele dividirse el texto en fragmentos.)
- **RAG (Retrieval-Augmented Generation)**: Se inyecta contexto externo (del manual tributario) dinámicamente en cada pregunta. Es decir, no entrenamos el modelo con el manual completo, sino que recuperamos *en tiempo real* los pasajes relevantes y se los damos al modelo en el prompt. OpenAI recomienda esta técnica para trabajos con documentos grandes【1†L91-L98】. Básicamente, la pregunta del usuario se enriquece con fragmentos recuperados de la base de conocimiento antes de pedir la respuesta.
- **Endpoints de la API**: Se pueden usar los endpoints de *Chat Completions* o bien la nueva *Responses API* (más orientada a flujos de agentes). En cualquiera, los tokens se cobran según el modelo: GPT-4.1 cuesta ~$2,00 por 1M tokens de entrada y $8,00 por 1M de salida【15†L74-L82】. Esto permite estimar costes por consulta (p.ej. ~0,01 USD por una consulta típica de pocos miles de tokens).
- **Embeddings**: Se recomienda usar la API de embeddings de OpenAI (e.g. `text-embedding-3-small`) para convertir texto a vectores semánticos【5†L158-L166】【33†L1-L4】. Estos embeddings se usan en la etapa de indexación (BD vectorial) del pipeline RAG.
- **Precisión y costos**: GPT-4.1 es muy preciso en instrucciones complejas y razonamiento, pero relativamente más caro. Sus variantes mini/nano son más económicas (~10-40% del costo) y bastante competentes en tareas simples【1†L79-L84】【15†L74-L82】. Se puede diseñar un *router*: preguntas sencillas o de alto volumen por GPT-4.1 mini, y las de alto detalle por GPT-4.1 normal. OpenAI incluso permite “caching” de prompts (preguntas repetidas) reduciendo el costo al 10%【30†L1-L4】.
- **Integración multimodal opcional**: GPT-4.1 es text-only. Si se quisieran analizar imágenes directas (diagramas, tablas), habría que usar GPT-4V o GPT-5 multimodal. Sin embargo, aquí compensamos haciendo OCR del texto en pantalla. En todo caso, el flujo RAG podría extenderse a embeddings de imágenes (p.ej. con CLIP) para soporte visual【20†L69-L78】.
- **Actualizaciones y entrenamiento**: OpenAI afirma que no entrena sus modelos con datos de clientes por defecto【17†L54-L63】. Sin embargo, dado que se consultan datos fiscales sensibles, es recomendable una instancia empresarial o controles para no retener información indebida.

## 3. Diseño del pipeline OCR → Embeddings → BD vectorial → RAG → API

1. **Extracción OCR**: La captura de pantalla se convierte a texto vía OCR. Se pueden usar bibliotecas como Tesseract (pytesseract) o EasyOCR【5†L86-L94】【5†L123-L131】. Es clave preprocesar la imagen (blanco y negro, umbral, rotación) para mejorar precisión【5†L137-L139】. El resultado es un bloque de texto con el contenido visible.
2. **Limpieza y segmentación**: El texto bruto se limpia (eliminar espacios/ruido) y se divide en fragmentos manejables. Por ejemplo, se puede usar un splitter recursivo (LangChain TextSplitter) para trocear en ~500–5000 tokens con cierto solapamiento (~300 tokens)【33†L1-L4】【32†L292-L300】. Esto preserva el contexto semántico a lo largo de los cortes.
3. **Cálculo de embeddings**: Cada fragmento de texto se convierte en un vector mediante un modelo de embedding (por ejemplo, OpenAI `text-embedding-3-small` o un modelo multilingüe equivalente). Esto ubica cada fragmento en el espacio vectorial semántico.
4. **Base de datos vectorial**: Los vectores se indexan en un motor de búsqueda vectorial (p.ej. **FAISS**, Chroma o Pinecone). Esto permite recuperar eficientemente los fragmentos más similares a cualquier consulta. Un ejemplo de código con FAISS se muestra en Shruti Shreya: una vez obtenidos los embeddings, se crea un índice FAISS con `FAISS.from_embeddings(embeddings, chunks)`【5†L158-L166】.
5. **Recuperación de contexto (RAG)**: Al llegar una pregunta del usuario, calculamos su embedding y buscamos en la BD los *k* fragmentos más cercanos. Estos pasajes (o su texto) conforman el **contexto relevante** para la pregunta.  
6. **Construcción del prompt**: El prompt a GPT-4.1 combina la pregunta con el contexto recuperado. Por ejemplo: 
   ```
   Eres un asesor fiscal experto. Usa SOLO este contexto para responder:

   [fragmento1]
   [fragmento2]
   …

   Pregunta: ¿Cómo tributa X?
   ```
   Este enfoque *groundea* la respuesta en la documentación real.
7. **Invocación del LLM**: Se envía el prompt construido a la API de OpenAI (GPT-4.1) mediante el endpoint elegido. El modelo genera la respuesta, que típicamente usa la terminología y normativas buscadas.
8. **Posprocesado y presentación**: La respuesta del LLM se formatea para el usuario. Puede incluir citas de las fuentes (p.ej. numeradas) si interesa, o resaltar documentos relevantes. Opcionalmente, se almacena la conversación o se cachea el prompt + respuesta para agilizar consultas similares en el futuro.
9. **RAG multimodal (opcional)**: Si hay imágenes (gráficos) en las capturas, se puede enriquecer el sistema con embeddings de imágenes usando CLIP【20†L69-L78】. En el artículo de ejemplo, se combinan embeddings de texto e imagen en un único FAISS, permitiendo recuperar fragmentos visuales y textuales【20†L69-L78】【20†L133-L142】. Luego se pueden incluir imágenes como “puntos de referencia” en el prompt a GPT-4.1, aunque esto es avanzado. 
10. **Ejemplo real de pipeline**: En un caso concreto, se ingieren PDFs de leyes y manuales, se fragmentan a 5000 tokens con overlap 300, se embeben con el modelo de OpenAI, se guardan en Chroma y luego se usa LangChain para consultas con GPT-4.1 (mini)【33†L1-L4】【32†L228-L236】. Este flujo RAG ha demostrado respuestas precisas en preguntas complejas, respetando las normas del contenido indexado.

## 4. Prototipo de interfaz (overlay, extensión o webapp)

- **Extensión de navegador**: Una opción natural es una extensión Chrome/Edge. En su manifest se pide permiso para capturar pestañas y ejecutar scripts. Al clic en el icono, se inyecta un *content script* que muestra un overlay (canvas) para seleccionar el área. Luego se captura la imagen (por ejemplo con `chrome.tabs.captureVisibleTab`) y se manda al OCR. La extensión puede entonces abrir un pequeño panel o popup donde muestra la respuesta. Un ejemplo existente es **ContextCapture**, que hace exactamente esto: el usuario hace clic en el icono, arrastra para seleccionar, y el plugin automáticamente toma esa región, extrae el texto con OCR y genera un resumen de 3 líneas con GPT【8†L293-L301】【8†L358-L366】. Este flujo de uso (botón → seleccionar área → OCR → AI) sirve de guía para nuestro diseño.
- **Aplicación web/PWA popup**: Otra alternativa es una página web o aplicación de escritorio (p.ej. con Electron o Tauri) que flote sobre otras ventanas. Podría tener un botón “Capturar pantalla” que llame a `getDisplayMedia` o similares para compartir la pantalla, o hacer una captura con canvas. Después se permitiría dibujar sobre la imagen para recortar la región. Esto ofrece flexibilidad gráfica (controles, historial, configuración) y se puede actualizar desde un servidor.
- **Aplicación local (Python)**: Como prototipo sencillo, se puede escribir una GUI en Python. Con **PySimpleGUI** (o Qt/Tkinter) se puede crear una ventana de interfaz que muestre controles (botón “Capturar”). Al pulsarlo, se despliega una ventana transparente (como vimos en [10]) para seleccionar área y luego PyAutoGUI captura ese rectángulo. Por ejemplo, PySimpleGUI oculta la ventana de selección antes de capturar para no bloquearla【19†L53-L59】. Esta interfaz puede luego procesar la imagen y mostrar la respuesta en la misma ventana o mediante notificaciones.
- **Selección de región**: La UI (en extensión o app) debe permitir dibujar la selección con el cursor. Por ejemplo, ContextCapture dice “Drag to select the region containing text”【8†L358-L366】. Lo mismo haría la app local: click y arrastrar, con soporte para mover/ajustar el rectángulo.
- **Configuración de parámetros**: La interfaz incluiría opciones como **escala de captura**, **idioma de OCR**, intervalo de captura automática, habilitar/deshabilitar sonido o notificaciones al obtener respuesta, etc.
- **Modo de interacción**: Se definirá si el asistente funciona modo “conversacional” (historial de diálogo) o pregunta-respuesta independiente. Idealmente recuerda contexto fiscal por sesión.
- **Demostración existente**: ContextCapture es una prueba de concepto similar: overlay para región, OCR (OCR.Space o Tesseract.js) y GPT/Claude para resumen【8†L293-L301】【8†L358-L366】. Podemos inspirarnos en su flujo de uso descrito (pasos 1-4 en [8]).

## 5. Despliegue, seguridad y costes operativos

- **Infraestructura y despliegue**: 
  - Si es **una extensión/webapp**, basta publicar en la tienda de extensiones o servidor (con HTTPS). El backend (vector DB, OCR) puede ser local o en la nube (p.ej. una instancia Docker con FAISS/Chroma y librerías OCR). 
  - Si es **app local**, se empaqueta (por ejemplo con PyInstaller) o se distribuye con instrucciones de instalación. Puede levantarse en cualquier PC con GPU/CPU decente.
  - La parte de **vector DB** puede correr en el mismo cliente o servidor. Por ejemplo, instalar FAISS/Chroma en Docker en la máquina del usuario o en servidor dedicado. Los embeddings (almacenamiento de vectores) suelen ocupar gigabytes según tamaño del manual.
  - **Mantenimiento**: El contenido (legal, manuales) puede actualizarse periódicamente (añadir nuevos documentos). Esto requeriría re-embebido y re-indexado (o incremental).
- **Seguridad y privacidad**: 
  - Al manejar información fiscal (datos financieros personales), es crítico cumplir normas como GDPR. **OpenAI** declara que “no entrena sus modelos con tus datos por defecto” y cifra los datos en tránsito/reposo【17†L54-L63】【17†L62-L64】. Pero conviene usar planes Empresariales o controles extras para no retener datos sensibles. 
  - Solo deben enviarse a la API los fragmentos de texto estrictamente necesarios (no capturas completas con datos personales innecesarios). La interfaz debería alertar al usuario sobre privacidad y permitir cancelar consultas.
  - **Leyes de privacidad**: OpenAI recibió una multa de €15 millones en Italia por violaciones de GDPR【18†L43-L51】. Esto subraya la importancia de informar al usuario qué datos se envían y de cumplir con la ley (p.ej. no enviar datos de menores o datos sensibles sin consentimiento).
  - El almacenamiento local de la base vectorial (vectores) también debe protegerse (cifrado en disco, acceso restringido).
- **Costes operativos**:
  - **OpenAI API**: GPT-4.1 se tarifaba ~$2 por millón tokens entrada, $8 por millón salida【15†L74-L82】. Por ejemplo, una consulta de ~2000 tokens in y 1000 tokens out costaría ~\$0.012. El modelo **mini** cuesta ~$0.40/$1.60 por millón (un 80% menos)【15†L74-L82】. 
  - **Optimización**: Se recomienda usar GPT-4.1 para preguntas complejas y GPT-4.1 mini para tareas de menor complejidad o *pre-procesamiento* (p. ej. filtrar preguntas o clasificación simple), ahorrando costos. Además, el *caching* de prompts frecuentes reduce el costo al 10% del input【30†L1-L4】.
  - **Hardware local**: Si decidimos no usar la API OpenAI, y en su lugar correr modelos locales (como DeepSeek 8B), haría falta una GPU potente para inferencia (p.ej. A100/H100). Un GPU en la nube puede costar desde ~$0.10/h hasta varios dólares (un H100 está sobre \$3/h【26†L542-L551】). Además, los modelos locales de 3B-8B requieren decenas de GB de RAM. Sin GPUs, la inferencia sería muy lenta.
  - **Base de datos y OCR**: FAISS o Chroma pueden correr en CPU, apenas añaden costo si se usa máquina existente. OCR (Tesseract) corre en CPU y es gratuito. Si se usan servicios en la nube (p.ej. Pinecone, OCR API) sí habría un coste por uso/almacenamiento.
  - **Resumen de costes**: El gasto principal será la API de GPT-4.1 por consulta. Por ejemplo, 100 preguntas/día (~3000 tokens totales cada una) serían ~$3/día o ~$90/mes. A esto sumarían los costos de hosting (si se usa servidor en la nube) o simplemente electricidad/hardware local. Usando mini/nano y caching se puede reducir significativamente.
  - **Balance**: Dependiendo del volumen, podría convenir un sistema híbrido: desarrollo y pruebas locales (Ollama, DeepSeek) vs producción en OpenAI API para fiabilidad. También revisar nuevos modelos (GPT-5.x, DeepSeek V3) por costo/precisión【15†L74-L82】.
  
**Fuentes:** Arquitectura RAG basada en OCR y embeddings【5†L142-L150】【33†L1-L4】, ejemplos de interfaz con captura y OCR【8†L293-L301】【10†L65-L74】, y documentación oficial de OpenAI sobre GPT-4.1 y privacidad【1†L50-L57】【17†L54-L63】【18†L43-L51】. Costes estimados según la última guía de precios【15†L74-L82】【30†L1-L4】.