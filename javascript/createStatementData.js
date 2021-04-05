export default function createStatementData(invoices, plays){
    const statementData = {} ;
    statementData.customer = invoices.customer;
    statementData.performances = invoices.performances.map(enrichPerformance);
    statementData.totalAmount = totalAmount(statementData);
    statementData.totalVolumeCredits = totalVolumeCredits(statementData);
    return statementData;
    
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
