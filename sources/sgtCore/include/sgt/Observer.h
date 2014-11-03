#ifndef SGT_OBSERVER
#define SGT_OBSERVER 1

#include <sgtcore.h>
#include <sgt/Referenced.h>

namespace sgt {

/** Observer base class for tracking when objects are unreferenced (their reference count goes to 0) and are being deleted.*/
class SGTCORE_EXPORT Observer
{
    public:
        Observer();
        virtual ~Observer();

        /** objectDeleted is called when the observed object is about to be deleted.  The observer will be automatically
        * removed from the observed object's observer set so there is no need for the objectDeleted implementation
        * to call removeObserver() on the observed object. */
        virtual void objectDeleted(void*) {}

};

/** Class used by sgt::Referenced to track the observers associated with it.*/
class SGTCORE_EXPORT ObserverSet : public sgt::Referenced
{
    public:

        ObserverSet(const Referenced* observedObject);

        Referenced* getObservedObject() { return _observedObject; }
        const Referenced* getObservedObject() const { return _observedObject; }

        /** "Lock" a Referenced object i.e., protect it from being deleted
          *  by incrementing its reference count.
          *
          * returns null if object doesn't exist anymore. */
        Referenced* addRefLock();

        inline boost::mutex* getObserverSetMutex() const { return &_mutex; }

        void addObserver(Observer* observer);
        void removeObserver(Observer* observer);

        void signalObjectDeleted(void* ptr);

        typedef std::set<Observer*> Observers;
        Observers& getObservers() { return _observers; }
        const Observers& getObservers() const { return _observers; }

    protected:

        ObserverSet(const ObserverSet& rhs): sgt::Referenced(rhs) {}
        ObserverSet& operator = (const ObserverSet& /*rhs*/) { return *this; }
        virtual ~ObserverSet();

        mutable boost::mutex            _mutex;
        Referenced*                     _observedObject;
        Observers                       _observers;
};

}

#endif
