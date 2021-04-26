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
 
class Franc {
    private int amount;
    Franc(int amount)
    {
        this.amount = amount;
    }
    Franc times(int multiplier)
    {
        return new Franc(amount * multiplier);
    }
    public boolean equals(Object object)
    {
        Franc franc = (Franc) object;
        return amount == franc.amount;
    }
}
