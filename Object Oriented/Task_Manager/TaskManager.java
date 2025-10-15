import java.util.*;

public class TaskManager
{
    static int ID = 0;
    public Map<Integer,Task> Tasks = new HashMap<>();

    public void addTask(String name, int priority)
    {
        Task t = new Task(name, priority);
        this.Tasks.put(ID, t);
        ID++;
    }

    public Task getTask(int ID)
    {
        return Tasks.get(ID);
    }

    public void listTasks()
    {
        for (Map.Entry<Integer, Task> entry : Tasks.entrySet())
        {
            Task temp = entry.getValue();
            System.out.println(temp.name + ":  "+ temp.name);
        }
    }

    

}