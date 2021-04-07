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
 
import org.junit.Test;

import static org.junit.Assert.*;
 
public class MoneyTest 
{
    @Test
    public void testMultiplication()
    {
        Dollar five = new Dollar(5);
        five.times(2);
        assertEquals(10, five.amount);
    }
}