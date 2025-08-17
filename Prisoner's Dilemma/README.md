# Iterated Prisoner’s Dilemma – Streak Variant  

What if **tit-for-tat** went on a bad Tinder date and came back with trust issues?  
That’s this project: the **Iterated Prisoner’s Dilemma** with **streaks, errors, and tournaments**.  

---

## Engine  

- **200 rounds per match**  
- Moves: `1 = Cooperate`, `-1 = Defect`  
- **Streak bonuses** → longer mutual cooperation = bigger payoffs  
- **Error chance** (default 2%) → “oops, I didn’t mean to betray you”  
- Returns `(history, scores)`  

---

## Agents  

Agents subclass `base_agent` and define `next_move(history)`.  

```python
class Agent(base_agent):
    def next_move(self, history):
        return 1 if random.randint(0,100) > 50 else -1
````

---

## 🏆 Tournament

* Add agent names to `list.csv`
* Runs **round robin**
* Totals scores

```python
print(round_robin(10))
```

---

## 📂 Layout

```
eval_engine.py     # Game engine
Round Robin.py      # Round robin logic
Base_Agent.py      # Abstract base agent
agents/            # Your custom agents
list.csv           # Agents to include
```

---

## 🚀 Add an Agent

1. Drop `my_agent.py` into `agents/`
2. Write `class Agent(base_agent)`
3. Add `my_agent` to `list.csv`

