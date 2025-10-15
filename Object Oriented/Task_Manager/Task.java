import java.util.*;

public class Task{

    public String name;
    public Integer id;
    
    public int priority;
    public double duration;

    public String getName()
    {
        return this.name;
    }

    public int getPriority()
    {
        return this.priority;
    }

    public double getDuration()
    {
        return this.duration;
    }

    public void setName(String name)
    {
        this.name = name;
    }

    public void setPriority(int priority)
    {
        this.priority = priority;
    }

    public Task(String name)
    {
        this.name = name;
        this.priority = 1;
    }

    public Task(String name, int priority)
    {
        this.name = name;
        this.priority = priority;
    }

    public static void main(String[] args) {
        
    }
}