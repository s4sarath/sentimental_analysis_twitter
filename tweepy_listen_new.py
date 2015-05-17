from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

access_token =         "80039568-RhZutoPYcksyDXBmzt2NKTFnJhCcW20iyWc21SpFW"

access_token_secret =   "Ze6IzW2IWyjFA14ozJmA8tY14Xo7OyPDsMe8oVf5Foz5G"

consumer_key =         "fRPHUdtJyddcMGSVyMvBdXXua"

consumer_secret = "Wi00tnu479VFRTxc5URAumQywnTm5iUDqyefCElEjEYMcYNUsl" 


def some_task():

    class StdOutListener(StreamListener):

        def on_data(self, data):

            with open('/Users/sarathrnair/Sarath_works/Cfloor/cfloor_flask/sfetched_tweets.txt','w') as tf:
                tf.write(data)
            return True

            # print data
            # return True

        def on_error(self, status):
            print status


    # if __name__ == '__main__':

        #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(track=['python']) 

if __name__ == '__main__':

    from time import sleep
    from threading import Thread


    

    snooziness = int(raw_input('Enter the amount of seconds you want to run this: '))
    t = Thread(target=some_task)  # run the some_task function in another
                                  # thread
    t.daemon = True               # Python will exit when the main thread
                                  # exits, even if this thread is still
                                  # running
    t.start()
    sleep(snooziness)



# Since this is the end of the script, Python will now exit.  If we
# still had any other non-daemon threads running, we wouldn't exit.
# However, since our task is a daemon thread, Python will exit even if
# it's still going.