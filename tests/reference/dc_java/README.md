# Java DC reference

The study's directional-change detection (`regime_fusion.dc`) is written in Python and checked against the group's earlier Java code: `generateEvents`, `detect2` and `adjust` from `misc/DCEventGenerator.java` in [adesolaadegboye/SymbolicRegression](https://github.com/adesolaadegboye/SymbolicRegression) (commit `af564a5`).

The fixtures here are the Java code's answers, so the tests need no Java:

- `series/`: five synthetic price series, made by `make_fixtures.py`;
- `events.csv`: the events the Java code found in them at five thresholds (0.5%, 1%, 2%, 5%, 10%).

They were produced once, on 28 October 2026, by `DCReference.java`, those three functions copied unchanged apart from a reduced `Event` class. The repository is Python only, so that file is kept in its history, not in the current tree. To regenerate the fixtures (only if the series change; needs a Java JDK):

```bash
git show 782f43e:tests/reference/dc_java/DCReference.java > tests/reference/dc_java/DCReference.java
python tests/reference/dc_java/make_fixtures.py
```

## Result (`tests/test_dc_java_crosscheck.py`)

- **Confirmation points match exactly** for all 6,798 events.
- **Start points (extremes) match for 3,825 events.** The other 2,973 differ in three edge cases of the Java code, each checked exactly by the test:
  1. An upturn with no lower low after the downturn's confirmation: the extreme is that confirmation point, but Java starts the upturn one bar later (1,254 events).
  2. A downturn with no higher high after the upturn's confirmation: Java reuses the start of the previous upturn, so the downturn appears to start before that upturn was confirmed (1,348 events). The cause is `adjust(..., indexOS, indexOS)` in the upturn branch, which passes the overshoot indices where the DC indices are expected.
  3. An upturn whose low comes exactly two bars after the downturn's confirmation: Java starts it one bar early (371 events).

Confirmation-based quantities therefore agree. Quantities built on start points, such as DC length, overshoot length and total move, differ in these cases.
