#ifndef SGT_REFERENCED
#define SGT_REFERENCED 1

#include <sgtcore.h>

namespace sgt {

// forward declare, declared after Referenced below.
// class DeleteHandler;
class Observer;
class ObserverSet;

/** Base class for providing reference counted objects.*/
class SGTCORE_EXPORT Referenced
{

    public:
        Referenced();

        explicit Referenced(bool threadSafeRefUnref);

        Referenced(const Referenced&);

        inline Referenced& operator = (const Referenced&) { return *this; }

        /** Set whether to use a mutex to ensure ref() and unref() are thread safe.*/
        virtual void setThreadSafeRefUnref(bool threadSafe);

        /** Get whether a mutex is used to ensure ref() and unref() are thread safe.*/

        bool getThreadSafeRefUnref() const { return true; }

        /** Get the mutex used to ensure thread safety of ref()/unref(). */
        boost::mutex* getRefMutex() const { return getGlobalReferencedMutex(); }

        /** Get the optional global Referenced mutex, this can be shared between all sgt::Referenced.*/
        static boost::mutex* getGlobalReferencedMutex();

        /** Increment the reference count by one, indicating that
            this object has another pointer which is referencing it.*/
        inline int ref() const;

        /** Decrement the reference count by one, indicating that
            a pointer to this object is no longer referencing it.  If the
            reference count goes to zero, it is assumed that this object
            is no longer referenced and is automatically deleted.*/
        inline int unref() const;

        /** Decrement the reference count by one, indicating that
            a pointer to this object is no longer referencing it.  However, do
            not delete it, even if ref count goes to 0.  Warning, unref_nodelete()
            should only be called if the user knows exactly who will
            be responsible for, one should prefer unref() over unref_nodelete()
            as the latter can lead to memory leaks.*/
        int unref_nodelete() const;

        /** Return the number of pointers currently referencing this object. */
        inline int referenceCount() const { return _refCount; }


        /** Get the ObserverSet if one is attached, otherwise return NULL.*/
        ObserverSet* getObserverSet() const
        {
            return static_cast<ObserverSet*>(_observerSet.load(boost::memory_order_relaxed));
        }

        /** Get the ObserverSet if one is attached, otherwise create an ObserverSet, attach it, then return this newly created ObserverSet.*/
        ObserverSet* getOrCreateObserverSet() const;

        /** Add a Observer that is observing this object, notify the Observer when this object gets deleted.*/
        void addObserver(Observer* observer) const;

        /** Remove Observer that is observing this object.*/
        void removeObserver(Observer* observer) const;

    public:

        /** Set whether reference counting should use a mutex for thread safe reference counting.*/
        static void setThreadSafeReferenceCounting(bool enableThreadSafeReferenceCounting);

        /** Get whether reference counting is active.*/
        static bool getThreadSafeReferenceCounting();

#if 0        
        friend class DeleteHandler;

        /** Set a DeleteHandler to which deletion of all referenced counted objects
          * will be delegated.*/
        static void setDeleteHandler(DeleteHandler* handler);

        /** Get a DeleteHandler.*/
        static DeleteHandler* getDeleteHandler();
#endif


    protected:

        virtual ~Referenced();

        void signalObserversAndDelete(bool signalDelete, bool doDelete) const;

        // void deleteUsingDeleteHandler() const;

        mutable boost::atomic<void*>   _observerSet;
        mutable boost::atomic<int>     _refCount;
};

inline int Referenced::ref() const
{
    return _refCount.fetch_add(1, boost::memory_order_relaxed)+1;
    // return ++_refCount;
}

inline int Referenced::unref() const
{
    int newRef;

    // newRef = --_refCount;
    newRef = _refCount.fetch_sub(1, boost::memory_order_release)-1;

    if (newRef == 0)
    {
        boost::atomic_thread_fence(boost::memory_order_acquire);
        signalObserversAndDelete(true,true);
    }
    return newRef;
}

// intrusive_ptr_add_ref and intrusive_ptr_release allow
// use of sgt Referenced classes with boost::intrusive_ptr
inline void intrusive_ptr_add_ref(Referenced* p) { p->ref(); }
inline void intrusive_ptr_release(Referenced* p) { p->unref(); }

}

#endif
