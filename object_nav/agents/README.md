# Agents

`object_nav.agents` contains simple action policies for Habitat ObjectNav
experiments. Agents expose the same small surface:

```python
agent.reset()
action = agent.act(obs)
```

`reset()` is called once per episode. `act(...)` returns a Habitat action name,
for example `move_forward`, `turn_left`, `look_up`, or `stop`.

## InteractiveKeyboardAgent

Use this when controlling the robot manually from the OpenCV display loop:

```python
from object_nav.agents import InteractiveKeyboardAgent

agent = InteractiveKeyboardAgent()
...
agent.reset()
...
action = agent.act(obs)
obs = env.step(action)
```

Default controls:

- `W`: `move_forward`
- `A`: `turn_left`
- `D`: `turn_right`
- up arrow: `look_up`
- down arrow: `look_down`
- `F`: `stop`

The agent uses `cv2.waitKeyEx`, so extended arrow-key events work across common
OpenCV GUI backends. An OpenCV window must be active for keyboard events to
arrive reliably. The active ObjectNav config defines both look actions with a
30-degree tilt.

## RandomActionAgent

Use this for quick smoke tests that do not need manual input:

```python
from object_nav.agents import RandomActionAgent

agent = RandomActionAgent()
```

By default it samples uniformly from forward/left/right actions and never
chooses `stop`. Pass a different `actions` sequence when needed.

## Adding Agents

Keep new agents small and compatible with the same `reset()` and `act(obs)`
shape. If an agent needs heavy optional dependencies, import them inside that
agent module rather than in `object_nav.agents.__init__`.
