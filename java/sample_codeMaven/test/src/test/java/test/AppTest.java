//package test;

//import static org.junit.Assert.assertTrue;

//import org.junit.Test;

/**
 * Unit test for simple App.
 */
//public class AppTest 
//{
    /**
     * Rigorous Test :-)
     */
//    @Test
//    public void shouldAnswerWithTrue()
//    {
//        assertTrue( true );
//    }
//}

//AppTest.java
package test;
 
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Test;
 
public class AppTest 
{
    private App app = new App();
    @Test
    public void shouldAnswerWithTrue()
    {
        String user = "user";
        String pass = "pass";
        boolean result = app.login(user, pass);
        assertTrue(result);
    }
    @Test
    public void shouldAnswerWithFalse()
    {
        String user = "foo";
        String pass = "bar";
        boolean result = app.login(user, pass);
        assertFalse(result);
    }
}