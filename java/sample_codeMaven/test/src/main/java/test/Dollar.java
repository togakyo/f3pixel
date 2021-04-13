//package test;

/**
 * Hello world!
 *
 */
//public class App 
//{
//    public static void main( String[] args )
//    {
//        System.out.println( "Hello World!" );
//    }
//}

//App.java
package test;
 
class Dollar {
    private int amount;
    Dollar(int amount)
    {
        this.amount = amount;
    }
    Dollar times(int multiplier)
    {
        return new Dollar(amount * multiplier);
    }
    public boolean equals(Object object)
    {
        Dollar dollar = (Dollar) object;
        return amount == dollar.amount;
    }
}
