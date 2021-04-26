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
 
class Money {
    protected int amount;
    public boolean equals(Object object)
    {
        Money money = (Money) object;
        return amount == money.amount;
    }
}
