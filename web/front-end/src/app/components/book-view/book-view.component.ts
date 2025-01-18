import { Component, numberAttribute, OnInit } from '@angular/core';
import { BookCardComponent } from "../book-card/book-card.component";
import { CommonModule } from '@angular/common';
import { BookService } from '../../book-service';
import { Book } from '../../book-service';
import { ActivatedRoute, Router } from '@angular/router';
import { UserService } from '../../user.service';
import { UserlikeService } from '../../userlike.service';

@Component({
  selector: 'app-book-view',
  imports: [BookCardComponent, CommonModule],
  templateUrl: './book-view.component.html',
  styleUrl: './book-view.component.css',
  standalone: true
})
export class BookViewComponent implements OnInit {
  lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
  data: Book[] = [];
  currentPage = 1;
  pageSize = 10;
  likedBooks = false;

  constructor(private bookService: BookService,
    private route: ActivatedRoute,
    private router: Router,
    private userService: UserService,
    private userLikedService: UserlikeService) { }

    ngOnInit() {
      if (this.router.url.startsWith('/search')) {
        this.route.queryParams.subscribe(params => {
          const categories = params['categories'] || null;
          const isbn = params['isbn'] || null;
          const title = params['title'] || null;
          const author = params['author'] || null;
          const publisher = params['publisher'] || null;
          const startYear = params['startYear'] || null;
          const endYear = params['endYear'] || null;

          this.bookService.searchBooks(
            isbn, title, author, publisher, categories,startYear, endYear, this.pageSize, this.currentPage
          ).subscribe(books => {
            this.data = books;
            console.log(this.data)
          })
          });
    } else {
      if (this.router.url.startsWith('/featured'))if (this.router.url.startsWith('/featured')) {
        const username = this.userService.username;
        if (username && this.userLikedService.getLikedBooks(username).length > 0) {
          this.likedBooks = true;
          const likedBooks = this.userLikedService.getLikedBooks(username);
          console.log(likedBooks);
          const likedBookIds = likedBooks.map(book => book.bookId);
          this.bookService.getRecommendations(likedBookIds).subscribe(books=> {
            this.data = books;
          })
          console.log('Get recommendations here');
        } else {
          this.likedBooks = false;
          console.log('Tell user to like books here');
        }
      }

  }
}
  onPageChange(pageNumber: number) {
    this.currentPage = pageNumber;
    if (this.router.url.startsWith('/search')) {
      this.route.queryParams.subscribe(params => {
        const categories = params['categories'] || null;
        const isbn = params['isbn'] || null;
        const title = params['title'] || null;
        const author = params['author'] || null;
        const publisher = params['publisher'] || null;
        const startYear = params['startYear'] || null;
        const endYear = params['endYear'] || null;

        this.bookService.searchBooks(
          isbn, title, author, publisher, categories,startYear, endYear, this.pageSize, this.currentPage
        ).subscribe(books => {
          this.data = books;
          console.log(this.data)
      });
    });
  } else if (this.router.url.startsWith('/featured')) {
    console.log("display next 10 books here");
  }
  }
}


