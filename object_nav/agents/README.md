# Agents

`object_nav.agents` contains simple action policies for Habitat ObjectNav
experiments. Agents expose the same small surface:

```python
agent.reset()
action = agent.act(obs)
```

`reset()` is called once per episode. `act(...)` returns a Habitat action name,
for example `move_forward`, `turn_left`, `turn_right`, or `stop`.

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
- `F`: `stop`

The agent uses `cv2.waitKey`, so an OpenCV window must be active for keyboard
events to arrive reliably.

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
