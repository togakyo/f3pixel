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
 
public class App {
    public boolean login(String user, String pass){
        if ("user".equals(user) && "pass".equals(pass)){
            return true;
        } else {
            return false;
        }
    }
}
