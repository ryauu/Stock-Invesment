import sqlite3
def sql(stock):    
    with sqlite3.connect("stock_invesment.db") as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS VIX(
                price Real,
                priceChange real,
                priceDayHigh real,
                priceDayLow real,
                created_at text
            )
        """)
        cur.execute("""
            INSERT INTO VIX(price,priceChange,priceDayHigh,priceDayLow,created_at)
            VALUES (?,?,?,?,?)
        """, (stock.price,stock.priceChange,stock.priceDayHigh,stock.priceDayLow,stock.ts)
        )
        cur.execute("""
        DELETE FROM VIX
        WHERE rowid not in(
            SELECT min(rowid)
            from VIX
            group by price,priceChange,priceDayHigh,priceDayLow,date(created_at)
        )
        """)