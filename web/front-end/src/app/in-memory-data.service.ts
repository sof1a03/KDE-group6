import { Injectable } from '@angular/core';
import { InMemoryDbService } from 'angular-in-memory-web-api';
import { RequestInfo } from 'angular-in-memory-web-api';
import { Book }  from './book-service';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class InMemoryDataService implements InMemoryDbService {
    books: Book[] = [
      {
          "name": "Flowers for Algernon",
          "url": "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg",
          "id": "0",
          "genres": ["Science Fiction", "Drama"],
          "publisher": "Harcourt, Brace & World",
          "year": 1966,
          "ISBN": "978-0-15-632940-9",
          "author": "Daniel Keyes"
      },
      {
          "name": "Lord of the Rings",
          "url": "https://m.media-amazon.com/images/I/913sMwNHsBL._SL1500_.jpg",
          "id": "1",
          "genres": ["Fantasy", "Adventure"],
          "publisher": "George Allen & Unwin",
          "year": 1954,
          "ISBN": "978-0-618-00221-3",
          "author": "J.R.R. Tolkien"
      },
      {
          "name": "The Perfume",
          "url": "https://upload.wikimedia.org/wikipedia/en/f/f5/PerfumeSuskind.jpg",
          "id": "2",
          "genres": ["Historical Fiction", "Thriller"],
          "publisher": "Diogenes Verlag",
          "year": 1985,
          "ISBN": "978-0-394-53982-5",
          "author": "Patrick Süskind"
      },
      {
          "name": "Fall or Dodge; in Hell",
          "url": "https://m.media-amazon.com/images/I/91Z1I-2LQpL._AC_UF1000,1000_QL80_.jpg",
          "id": "3",
          "genres": ["Science Fiction", "Satire"],
          "publisher": "Simon & Schuster",
          "year": 2019,
          "ISBN": "978-1-4767-3802-9",
          "author": "Neal Stephenson"
      },
      {
          "name": "Brave New World",
          "url": "https://m.media-amazon.com/images/I/91D4YvdC0dL._AC_UF894,1000_QL80_.jpg",
          "id": "4",
          "genres": ["Science Fiction", "Dystopian"],
          "publisher": "Chatto & Windus",
          "year": 1932,
          "ISBN": "978-0-06-085052-4",
          "author": "Aldous Huxley"
      },
      {
          "name": "The Midnight Library",
          "url": "https://media.s-bol.com/yyE0mpJQW856/o2jMxV3/547x840.jpg",
          "id": "5",
          "genres": ["Fantasy", "Magical Realism"],
          "publisher": "Canongate Books",
          "year": 2020,
          "ISBN": "978-1-78689-273-5",
          "author": "Matt Haig"
      },
      {
          "name": "Cryptonomicon",
          "url": "https://m.media-amazon.com/images/I/811HmmUbx3L._AC_UF1000,1000_QL80_.jpg",
          "id": "6",
          "genres": ["Historical Fiction", "Science Fiction"],
          "publisher": "Spectra Books",
          "year": 1999,
          "ISBN": "978-0-380-78862-0",
          "author": "Neal Stephenson"
      },
      {
          "name": "Hersenschimmen",
          "url": "https://media.s-bol.com/vZg12A3m36QM/739x1200.jpg",
          "id": "7",
          "genres": ["Psychological Fiction", "Drama"],
          "publisher": "De Bezige Bij",
          "year": 1984,
          "ISBN": "978-90-234-5363-9",
          "author": "J. Bernlef"
      },
      {
          "name": "Tokyo Vice",
          "url": "https://m.media-amazon.com/images/I/61TWK--pVVL._AC_UF894,1000_QL80_.jpg",
          "id": "8",
          "genres": ["True Crime", "Memoir"],
          "publisher": "Pantheon Books",
          "year": 2009,
          "ISBN": "978-0-307-37879-0",
          "author": "Jake Adelstein"
      }
  ];
  createDb() {
        return {'books':this.books};
    }
  // Implement `get` to handle both collections and individual item requests

  get(reqInfo: RequestInfo): Observable<any> {
    const { collectionName, id, query, req } = reqInfo;

    if (collectionName === 'books') {
      if (id) {
        // If an ID is provided, find and return the specific book
        const item = this.books.find(book => book.id === id);
        return reqInfo.utils.createResponse$(() => ({
          body: item,
          status: item ? 200 : 404
        }));
      } else {
        // If no ID is provided, return all books
    // Return a default response if the collection is not handled

    let books = [...this.books]; // Create a copy of the books array


    // Apply filtering if query parameters are present
    if (query?.has('page') || query?.has('pageSize')) {
      const page = +(query?.get('page') ?? '1'); // Use ?? here
      const pageSize = +(query?.get('pageSize') ?? '10'); // Use ?? here

      const startIndex = (page - 1) * pageSize;
      const endIndex = startIndex + pageSize;

      books = books.slice(startIndex, endIndex);
    }

    return reqInfo.utils.createResponse$(() => ({
      body: books,
      status: 200,
    }));
  }

        return reqInfo.utils.createResponse$(() => ({
          body: this.books,
          status: 200
        }));
      }


    return reqInfo.utils.createResponse$(() => ({
      body: [],
      status: 404
    }));
  }
}
