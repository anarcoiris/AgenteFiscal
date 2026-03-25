# Antigravity Architecture - Charlie Agent

Esta documentación refleja el estado de la arquitectura **Antigravity** migrada desde el monolito `CharlieAgent` v5.

## Estado Actual (Logros Completados)
- **Separación de capas completada** según las 6 fases de `Target.md`:
  - `charlie_perception.py`: Capa de percepción separada (DOM y OS).
  - `charlie_classifier.py`: Motor de clasificación que decide si una tarea va a `browser`, `desktop` o `hybrid`.
  - `charlie_desktop_controller.py`: Stub funcional del controlador de entorno OS (usando `pyautogui`, `mss`, `pillow`).
  - `charlie_orchestrator.py`: Reemplazo del entrypoint principal que coordina Percepción -> Clasificación -> Planificación -> Controlador.
- **Modelos de datos**: Definidos en `charlie_models.py` para sesiones, trazas, átomos UI y memoria de skills.
- **Integración UI**: `charlie_dashboard.py` consume ahora `CharlieOrchestrator` en lugar de `CharlieAgent`.

## Problemas / Limitaciones Actuales
- La ejecución web en `charlie_orchestrator.py` todavía utiliza el `CharlieAgent` completo como fallback internamente para no romper compatibilidad. Falta la purga total del agente monolítico.
- El `DesktopController` no tiene todavía un bucle de autocorrección visual (healing) completo como el sistema web.
- **Limitaciones de las dependencias actuales del SO**: Actualmente se usa Python puro (`pyautogui`, `mss`), que añade un sobrecoste de latencia en la iteración visual y de teclado. 

## Dirección Estratégica Híbrida (OS-Level Control)
Para lograr el máximo de rendimiento, baja latencia y eludir restricciones de ciertas GUIs de Windows, el plan a largo plazo para interactuar con el sistema operativo es abandonar `pyautogui` y emplear rutinas escritas en **C/C++**. Alternativamente (y de forma muy pythónica) podemos utilizar la librería nativa **`ctypes`**, enlazando directamente al API de Windows `user32.dll` (ej. `SendInput`, `mouse_event`, `BitBlt` para lectura rápida de memoria de video) para obtener el mismo rendimiento sin requerir compiladores adicionales en las máquinas de los usuarios.

## Cambios Pendientes (Hacia Adelante)
1. **Refactor Browser Controller**: Deprecar `charlie_agent.py` y extraer solo su motor de interpretación a un `charlie_browser_controller.py` puro (que implemente la misma abstracción que el desktop).
2. **Implementar Skills Engine**: Lógica que lea `charlie_models.Skill` desde la memoria episódica y decida autocompletar una tarea sin llamar al LLM si la confianza es total.
3. **Ruteo de Flujo Híbrido Real**: Validar una tarea de ida y vuelta (ej. "Abre Excel en Desktop, lee un valor, y rellénalo en Infojobs").
