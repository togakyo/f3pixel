package test;
 
class Franc extends Money{
    Franc(int amount, String currency)
    {
        super(amount, currency);
    }
    Money times(int multiplier)
    {
        return Mpney.franc(amount * multiplier);
    }
}
