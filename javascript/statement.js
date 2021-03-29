
var invoices = '{"customer": "BigCo","performances": [{"playID": "hamlet","audience": 55},{"playID": "as-like","audience": 35},{"playID": "othello","audience": 40}]}';
var plays = '{"hamlet":{"name": "Hamlet", "type": "tragedy"},"as-like":{"name": "As You Like It", "type": "comedy"},"othello": {"name": "Othello", "type": "tragedy"}}';

var jsonObject_invoices = JSON.parse(invoices);　　　//HACK: invoices JSON読み込みを想定
var jsonObject_plays = JSON.parse(plays);　　　      //HACK: plays JSON読み込みを想定

function statement (invoices, plays){
    
    let totalAmount = 0;
    let result =  " Statement for "+ invoices.customer + "\n"; //'Statement for ${invoices.customer}¥n';

    function playFor(aPerformance){
        return plays[aPerformance.playID];
    }

    function volumeCreditsFor(ePerformance){
        let result = 0;
        result += Math.max(ePerformance.audience - 30, 0) ;
        if ("comedy" === playFor(ePerformance).type) result += Math.floor(ePerformance.audience / 5) ; 
        return result;
    }

    function usd(eNumber){
        return new Intl.NumberFormat("en-US",{ style: "currency", currency: "USD", minimumIntegerDigits: 2 }).format(eNumber/100);
    }
    function amountFor(aPerformance){
        let result = 0;
    
        switch (playFor(aPerformance).type){
            case "tragedy":
                result = 40000;
                if(aPerformance.audience > 30){
                    result += 1000 * (aPerformance.audience -30);
                }
                break;
            case "comedy":
                result = 30000;
                if(aPerformance.audience > 20){
                    result += 10000 + 500 * (aPerformance.audience -20);
                }
                result += 300 * aPerformance.audience;
                break;
            default:
                throw new Error("unknown type:" + playFor(aPerformance).type + "\n");
        }
    
        return result;
    }

    function totalVolumeCredits(){
        let volumeCredits = 0 ;
        for (let perf of invoices.performances) {
            volumeCredits += volumeCreditsFor(perf);
        }
    }


    for (let perf of invoices.performances) {
        result += "   "+playFor(perf).name+":" + usd(amountFor (perf)) + " " + (perf.audience) + " " + "seats\n";
        totalAmount += amountFor (perf);
    }

    result += " Amount owed is " + usd(totalAmount) + "\n";
    result += " You earned " + totalVolumeCredits() + " " +  "credits\n";
    return result ;
}



let test = statement(jsonObject_invoices, jsonObject_plays);
console.log(test);

//invoices = '/Users/togashikyousuke/Desktop/f3pixel/javascript/invoices.json';
//plays = '/Users/togashikyousuke/Desktop/f3pixel/javascript/plays.json';

//console.log(jsonObject_invoices.performances);
//console.log(jsonObject_plays.hamlet.name);
//console.log('Hello, World')

