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
 
class Franc extends Money{
    Franc(int amount)
    {
        this.amount = amount;
    }
    Money times(int multiplier)
    {
        return new Franc(amount * multiplier);
    }
}
