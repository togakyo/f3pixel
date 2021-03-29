
var invoices = '{"customer": "BigCo","performances": [{"playID": "hamlet","audience": 55},{"playID": "as-like","audience": 35},{"playID": "othello","audience": 40}]}';
var plays = '{"hamlet":{"name": "Hamlet", "type": "tragedy"},"as-like":{"name": "As You Like It", "type": "comedy"},"othello": {"name": "Othello", "type": "tragedy"}}';

var jsonObject_invoices = JSON.parse(invoices);　　　//HACK: invoices JSON読み込みを想定
var jsonObject_plays = JSON.parse(plays);　　　      //HACK: plays JSON読み込みを想定

function statement (invoices, plays){
    return renderPlainText(createStatementData(invoices, plays));

    function createStatementData(invoices, plays){
        const statementData = {} ;
        statementData.customer = invoices.customer;
        statementData.performances = invoices.performances.map(enrichPerformance);
        statementData.totalAmount = totalAmount(statementData);
        statementData.totalVolumeCredits = totalVolumeCredits(statementData);
        return statementData;    
    }

    function enrichPerformance(aPerformance){
        const result = Object.assign({}, aPerformance);
        result.play = playFor(result);
        result.amount = amountFor(result);
        result.volumeCredits = volumeCreditsFor(result);
        return result;
    }

    function playFor(aPerformance){
        return plays[aPerformance.playID];
    }

    function amountFor(aPerformance){
        let result = 0;
    
        switch (aPerformance.play.type){
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
                throw new Error("unknown type:" + aPerformance.play.type + "\n");
        }
    
        return result;
    }

    function volumeCreditsFor(ePerformance){
        let result = 0;
        result += Math.max(ePerformance.audience - 30, 0) ;
        if ("comedy" === ePerformance.play.type) result += Math.floor(ePerformance.audience / 5) ; 
        return result;
    }

    function totalAmount(data){
        let result = 0;
        for (let perf of data.performances) {
            result += perf.amount;
        }
        return result;
    }
    function totalVolumeCredits(data){
        let result = 0 ;
        for (let perf of data.performances) {
            result += perf.volumeCredits;
        }
        return result;
    }

}

function renderPlainText(data, plays){
    let result =  " Statement for "+ data.customer + "\n"; //'Statement for ${invoices.customer}¥n';
    for (let perf of data.performances){
        result += "  "+ perf.play.name+ ": " + usd(perf.amount) + " " + perf.audience + "seats" + "\n" ;
    }

    result += " Amount owed is " + usd(data.totalAmount) + "\n";
    result += " You earned " + data.totalVolumeCredits + " " +  "credits\n";
    return result ;

    function usd(eNumber){
        return new Intl.NumberFormat("en-US",{ style: "currency", currency: "USD", minimumIntegerDigits: 2 }).format(eNumber/100);
    }
}



let test = statement(jsonObject_invoices, jsonObject_plays);
console.log(test);

//invoices = '/Users/togashikyousuke/Desktop/f3pixel/javascript/invoices.json';
//plays = '/Users/togashikyousuke/Desktop/f3pixel/javascript/plays.json';

//console.log(jsonObject_invoices.performances);
//console.log(jsonObject_plays.hamlet.name);
//console.log('Hello, World')

